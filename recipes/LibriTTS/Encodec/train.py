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
from hyperpyyaml import load_hyperpyyaml
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.utils.logger import get_logger
from speechbrain.utils.text_to_sequence import text_to_sequence

logger = get_logger(__name__)


class EncodecBrain(sb.Brain):
    """The Brain implementation for Encodec"""

    def trim_prediction_to_match_input_audio(self, x, size: int):
        return x.narrow(2,0,size)
    
    def reconstruction_loss(self, x, y, eps = 1e-7):
        time_domain_loss = F.l1_loss(x, y)
        
        mel_x = hparams["mel_spectogram"](audio=x)
        mel_y = hparams["mel_spectogram"](audio=y)
        
        freq_domain_loss = (mel_x - mel_y).abs().mean()

        return time_domain_loss+ freq_domain_loss
    
    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss
    
    def compute_forward(self, batch, stage):
        """
        Computes the forward pass

        Arguments
        ---------
            batch: str
                a single batch
            stage: speechbrain.Stage
                the training stage

            Returns
        -------
        the model output
        """
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.audio
        
        encoder = self.modules.encoder
        quantizer = self.modules.quantizer
        decoder = self.modules.decoder
        
        latents = encoder(wavs)
        y = decoder(latents)

        return y, wav_lens
    
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
        mel_spec = batch.mel_spec

        y, _ = predictions
        
        y_narrowed = (self.trim_prediction_to_match_input_audio(y, batch.audio.data.shape[-1]))

        # We have original mel_spectro and generate mel_spectro
        loss = F.mse_loss(y_narrowed, batch.audio[0])  #self.feature_loss(batch.audio.data, y)#y_mel_spec.detach(),mel_spec)
        
        return self.reconstruction_loss(y_narrowed, batch.audio[0])

    def init_optimizers(self):
        
        self.optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )
        
    
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
