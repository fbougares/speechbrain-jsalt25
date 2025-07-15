# -*- coding: utf-8 -*-
"""
Recipe for training Encodec Neural Encoder from META
neural text-to-speech (TTS) system


Authors

"""
import os
import sys

import torch
import torchaudio
from collections import defaultdict
from hyperpyyaml import load_hyperpyyaml
import torch.nn.functional as F

import speechbrain as sb

from speechbrain.lobes.models.Encodec import EncodecModel
from speechbrain.lobes.models.Encodec_losses import total_loss, disc_loss

from speechbrain.utils.logger import get_logger
from speechbrain.utils.text_to_sequence import text_to_sequence
from speechbrain.utils.data_utils import scalarize



logger = get_logger(__name__)


class EncodecBrain(sb.Brain):
    """The Brain implementation for Encodec"""

    def trim_prediction_to_match_input_audio(self, x, size: int):
        return x.narrow(2,0,size)
    
    def compute_forward(self, batch, stage):
        """
        The forward function, generates synthesized waveforms,
        calculates the scores and the features of the discriminator for real and synthesized waveforms.

        Arguments
        ---------
            batch: str
                a single batch
            stage: speechbrain.Stage
                the training stage

        Returns 
        -------
        Generator audio output : output
        Quantizer loss : loss_w 
        Generator output : logits_real / fmap_real 
        Discriminator outputs : logits_fake/ fmap_fake.
        """
        
        batch = batch.to(self.device)
        input_wavs, wav_lens = batch.audio
        
        encoder = self.modules.encoder
        quantizer = self.modules.quantizer
        decoder = self.modules.decoder
        disc_model = self.modules.disc_model
        
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        sample_rate = 24_000
        channels = 1
        
        
        # putting all together : encoder - quantizer - decoder 
        model = EncodecModel(encoder, decoder, quantizer, target_bandwidths, sample_rate, channels)
        
        output, loss_w, _ = model(input_wavs)
        
        ### Doc : loss_w is the sum of all quantizer forward loss (RVQ commitment loss :l_w)
        logits_real, fmap_real = disc_model(input_wavs)
        logits_fake, fmap_fake = disc_model(output.detach()) # detach to avoid backpropagation to model
        
        return output, loss_w, logits_real, fmap_real, logits_fake, fmap_fake, sample_rate
    
    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss given the predicted (wav form) and target (wav form)

        Arguments
        ---------
        predictions : torch.Tensor
            The model generated wav form and other metrics from `compute_forward`
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for back-propagating the gradient
        """
        
        batch = batch.to(self.device)
        input_wavs, wav_lens = batch.audio
        
        (
            
            y_hat,
            loss_w,    # This is the sum of all quantizer forward losses (RVQ commitment loss :l_w)
            logits_real,
            fmap_real,
            logits_fake,
            fmap_fake,
            sample_rate,
            
        ) = predictions
        
        
        
        ### LOSS GENERATOR ########
        
        # Generator Losses (model waudio prediction) total loss of fmap (mel spec) and logits 
        losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                input_wavs, 
                y_hat, 
                sample_rate=sample_rate,
            ) 
        
        # without balancer: loss = 3*l_g + 3*l_feat + (l_t / 10) + l_f
        # loss_g = torch.tensor([0.0], device='cuda', requires_grad=True)
        loss_g = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f'] 
        
        
        ### LOSS DISCRIMINATOR  ##
        loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
        
        self.accumulated_loss_g += loss_g.item()
        for k, l in losses_g.items():
            self.accumulated_losses_g[k] += l.item()
        
        self.accumulated_loss_w += loss_w.item()
        
        self.accumulated_loss_disc += loss_disc.item()
        
        self.last_loss_stats[stage] = {"accumulated_loss_w": self.accumulated_loss_w,
                                       "loss_G": self.accumulated_loss_g,
                                       "loss_disc":self.accumulated_loss_disc}
        
        loss = {**loss_g, **loss_w, **loss_disc}
        
        return loss

    def on_stage_start(self, stage, epoch=None):
        """
        Gets called at the beginning of each epoch.

        Args:
            stage : sb.Stage / One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
            epoch (int, optional): The currently-starting epoch. This is passed `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.last_loss_stats = {}
        
    def on_stage_end(self, stage, stage_loss, epoch):
        """
        Gets called at the end of an epoch
        stage : sb.Stage / One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        stage_loss : float / The average loss for all of the data processed in this stage.
        epoch : int / The currently-starting epoch. This is passed `None` during the test stage.
        
        """
        
        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            lr = self.optimizer.param_groups[-1]["lr"]
            lr_disc = self.optimizer_disc.param_groups[-1]["lr"]
            self.last_epoch = epoch
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                                                stats_meta={"Epoch": epoch, "lr": lr, "lr_disc": lr_disc},
                                                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                                                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
        

    def init_optimizers(self):
        """
        Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """        
        self.optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        
        self.optimizer_disc = self.hparams.disc_opt_class(
            self.hparams.disc_model.parameters()
        )
        
        self.optimizers_dict = {
                "optimizer": self.optimizer,
                "optimizer_disc": self.optimizer_disc,
            }
        
        if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "optimizer", self.optimizer
                )
                self.checkpointer.add_recoverable(
                    "optimizer_disc", self.optimizer_disc
                )

    def on_fit_start(self):
        """
        Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics.
        """
        # Initialize variables to accumulate losses  
        self.accumulated_loss_g = 0.0
        self.accumulated_losses_g = defaultdict(float)
        self.accumulated_loss_w = 0.0
        self.accumulated_loss_disc = 0.0
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        
        
        
        return super().on_fit_start()
    
############### The data preparation method ########################
def dataio_prepare(hparams):
    # Define audio pipeline:

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("mel_spec", "audio") #  mel_spec is 80 dim / sig dim is 16khz or 24khz
    def audio_pipeline(wav):

        audio, sig_sr = torchaudio.load(wav)
        
        #audio = F.pad( audio, (0, (24000 - audio.size(-1))), "constant" )
        if sig_sr != hparams["sample_rate"]:
            audio = torchaudio.functional.resample(
                audio, sig_sr, hparams["sample_rate"]
            )
        print("Audio shape read ", audio.shape)
        mel_spec = hparams["mel_spectogram"](audio=audio.squeeze())


        return mel_spec, audio

    datasets = {}
    data_info = {
        "train": hparams["train_csv"],
        "valid": hparams["valid_csv"],
        "test": hparams["test_csv"],
    }
    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["mel_spec", "id", "audio"],
        )

        datasets[dataset] = datasets[dataset].filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )

    return datasets       
        

if __name__ == '__main__':
    
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    ### Fethi sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    
        # Prepare data
    if not hparams["skip_prep"]:
        sys.path.append("../../")
        from libritts_prepare import prepare_libritts

        sb.utils.distributed.run_on_main(
            prepare_libritts,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_csv_train": hparams["train_json"],
                "save_csv_valid": hparams["valid_json"],
                "save_csv_test": hparams["test_json"],
                "sample_rate": hparams["sample_rate"],
                "train_split": hparams["train_split"],
                "valid_split": hparams["valid_split"],
                "test_split": hparams["test_split"],
                "seed": hparams["seed"],
                "model_name": hparams["model"].__class__.__name__,
            },
        )
    
    datasets = dataio_prepare(hparams)
    
    # Brain class initialization
    encodec_brain = EncodecBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # Training
    encodec_brain.fit(
        encodec_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    
    # # Test
    # if "test" in datasets:
    #     encodec_brain.evaluate(
    #         datasets["test"],
    #         test_loader_kwargs=hparams["test_dataloader_opts"],
    #     )
