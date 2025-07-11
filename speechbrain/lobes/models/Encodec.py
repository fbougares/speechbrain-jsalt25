"""

The goal of this code is to integrate "EnCodec: High Fidelity Neural Audio Compression"  from Meta inside SpeechBrain



Authors
 * Fethi Bougares, 2025

"""

import math
import torch
import einops
import warnings
import numpy as np
import typing as tp
from torch import nn
from pathlib import Path
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm


from speechbrain.lobes.models.Encodec_utils import SEANetResnetBlock, SConv1d, SLSTM, SConvTranspose1d, LMModel, _check_checksum, _get_checkpoint_url
from speechbrain.quantization.vq import ResidualVectorQuantizer

import logging

logging.getLogger().setLevel(logging.INFO)

ROOT_URL = 'https://dl.fbaipublicfiles.com/encodec/v0/'
EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

###################################  CREATE THE EnCodec Encoder  ############################################
############## This will gives the output embedding Z from the Encoder before the QT module #################

class EncodecEncoder(nn.Module):
    """
        Encodec Encoder is a "Sound EnhAncement Network" published by Google in https://arxiv.org/pdf/2009.02095
        This class is the Encoder part which is the same architecture as the SEANet (Sound EnhAncement Network) encoder.
        
        Given a tensor `x`, returns a list of frames containing the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.
        Each frames is a tuple `(codebook, scale)`, with `codebook` of shape `[B, K, T]`, with `K` the number of codebooks.
        
        
        Args:
            channels (int): Audio channels.
            dimension (int): Intermediate representation dimension.
            n_filters (int): Base width for the model.
            n_residual_layers (int): nb of residual layers.
            ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
                                    that must match the decoder order.
            activation (str): Activation function.
            activation_params (dict): Parameters to provide to the activation function
            norm (str): Normalization method (function).
            norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
            kernel_size (int): Kernel size for the initial convolution.
            last_kernel_size (int): Kernel size for the last convolution.
            residual_kernel_size (int): Kernel size for the residual layers.
            dilation_base (int): How much to increase the dilation with each layer.
            causal (bool): Whether to use fully causal convolution.
            pad_mode (str): Padding mode for the convolutions.
            true_skip (bool): Whether to use true skip connection or a simple (streamable) convolution as the skip connection in the residual network blocks.
            compress (int): Reduced dimensionality in residual branches (from Demucs v3).
            lstm (int): Number of LSTM layers at the end of the encoder.    
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7, 
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False, 
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios ## why delete this !!
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        
        act = getattr(nn, activation)
        mult = 1 
        
        # Create a list of nn.Module called model 
        # Add the first SConv1d  convolutional layer -> local feature learning
        logging.info(f"First block in the Encoder")
        model: tp.List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        logging.info(f"Down Sampling blocks in the Encoder {self.ratios}")
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            logging.info(f"Residual Layers Encoder number {i}")
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)
                    ]

            # Add downsampling layers
            logging.info(f"Down Sampling Layers Encoder number {i}")
            model += [
                act(**activation_params),
                SConv1d(mult * n_filters, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2
        
        # Adding two LSTM layers on top of the convolution blocks 
        if lstm:
            logging.info(f"Adding {lstm} LSTM Layers Encoder number")
            model += [SLSTM(mult * n_filters, num_layers=lstm)]
        
        # Adding the final Conv 1D layer 
        logging.info(f"Adding last Conv 1D Layer" )
        model += [
            act(**activation_params),
            SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model) # serialize the model into a Sequantial container to apply forward function of each element in the model
        
        
    # Forward pass inside the encoder this will generate 
    def forward(self, x):
        return self.model(x)
    

##############################################  Encodec Decoder ################################################
######################### This will take the encoder outputs and go into the decoder blocks to generate audio segments #################

class EncodecDecoder(nn.Module):
    """

    This is a SEANET like decoder. 
    Encodec Decoder is a "Sound EnhAncement Network" published by Google in https://arxiv.org/pdf/2009.02095
        This class is the Decoder part which is the same architecture as the SEANet (Sound EnhAncement Network) Decoder.
        
    Args:
    channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
        
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2,
                 trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add upsampling layers
            model += [
                act(**activation_params),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y



################################ Encodec Encoder - Decoder #################################@@@
#TODO

class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
    
    def __init__(self, encoder: EncodecEncoder, decoder: EncodecDecoder, quantizer: ResidualVectorQuantizer, target_bandwidths: tp.List[float],
                 sample_rate: int, channels: int, normalize: bool = False, segment: tp.Optional[float] = None, overlap: float = 0.01, 
                 name: str = 'unset'):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios)) #75
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be a power of 2."


    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))    


    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length 
        if segment_length is None: #segment_length = 1*sample_rate
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride): # shift windows to choose data
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames


    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1] # tensor_cut or original
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x) # [2,1,10000] -> [2,128,32]
        #TODO: Encodec Trainer的training
        if self.training:
            return emb,scale
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes, scale    

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        if self.training:
            emb = codes
        else:
            codes = codes.transpose(0, 1)
            emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.encode(x) # input_wav -> encoder , x.shape = [BatchSize,channel,tensor_cut or original length] 2,1,10000
        if self.training:
            # if encodec is training, input_wav -> encoder -> quantizer forward -> decode
            loss_w = torch.tensor([0.0], device=x.device, requires_grad=True)
            codes = []
            # self.quantizer.train(self.training)
            index = torch.tensor(random.randint(0,len(self.target_bandwidths)-1),device=x.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(index, src=0)
            bw = self.target_bandwidths[index.item()]# fixme: variable bandwidth training, if you broadcast bd, the broadcast will encounter error
            for emb,scale in frames:
                qv = self.quantizer(emb,self.frame_rate,bw)
                loss_w = loss_w + qv.penalty # loss_w is the sum of all quantizer forward loss (RVQ commitment loss :l_w)
                codes.append((qv.quantized,scale))
            return self.decode(codes)[:,:,:x.shape[-1]],loss_w,frames
        else:
            # if encodec is not training, input_wav -> encoder -> quantizer encode -> decode
            return self.decode(frames)[:, :, :x.shape[-1]]

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.target_bandwidths}.")
        self.bandwidth = bandwidth

    def get_lm_model(self) -> LMModel:
        """Return the associated LM model to improve the compression rate.
        """
        device = next(self.parameters()).device
        lm = LMModel(self.quantizer.n_q, self.quantizer.bins, num_layers=5, dim=200,
                     past_context=int(3.5 * self.frame_rate)).to(device)
        checkpoints = {
            'encodec_24khz': 'encodec_lm_24khz-1608e3c0.th',
            'encodec_48khz': 'encodec_lm_48khz-7add9fc3.th',
        }
        try:
            checkpoint_name = checkpoints[self.name]
        except KeyError:
            raise RuntimeError("No LM pre-trained for the current Encodec model.")
        url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
        state = torch.hub.load_state_dict_from_url(
            url, map_location='cpu', check_hash=True)  # type: ignore
        lm.load_state_dict(state)
        lm.eval()
        return lm

    @staticmethod
    def _get_model(target_bandwidths: tp.List[float],
                   sample_rate: int = 24_000,
                   channels: int = 1,
                   causal: bool = True,
                   model_norm: str = 'weight_norm',
                   audio_normalize: bool = False,
                   segment: tp.Optional[float] = None,
                   name: str = 'unset',
                   ratios=[8, 5, 4, 2]):
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10)) # int(1000*24//(math.ceil(24000/320)*10))
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model

    @staticmethod
    def _get_pretrained(checkpoint_name: str, repository: tp.Optional[Path] = None):
        if repository is not None:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            file = repository / checkpoint_name
            checksum = file.stem.split('-')[1]
            _check_checksum(file, checksum)
            return torch.load(file)
        else:
            url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
            return torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)  # type:ignore

    @staticmethod
    def encodec_model_24khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained causal 24khz model.
        """
        if repository:
            assert pretrained
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        checkpoint_name = 'encodec_24khz-d7cc33bc.th'
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
            target_bandwidths, sample_rate, channels,
            causal=True, model_norm='weight_norm', audio_normalize=False,
            name='encodec_24khz' if pretrained else 'unset')
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def encodec_model_48khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained 48khz model.
        """
        if repository:
            assert pretrained
        target_bandwidths = [3., 6., 12., 24.]
        checkpoint_name = 'encodec_48khz-7e698e3e.th'
        sample_rate = 48_000
        channels = 2
        model = EncodecModel._get_model(
            target_bandwidths, sample_rate, channels,
            causal=False, model_norm='time_group_norm', audio_normalize=True,
            segment=1., name='encodec_48khz' if pretrained else 'unset')
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def my_encodec_model(checkpoint: str,ratios=[8,5,4,2]):
        """Return the pretrained 24khz model.
        """
        import os
        assert os.path.exists(checkpoint), "checkpoint not exists"
        print("loading model from: ",checkpoint)
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
                target_bandwidths, sample_rate, channels,
                causal=False, model_norm='time_group_norm', audio_normalize=True,
                segment=None, name='my_encodec',ratios=ratios)
        pre_dic = torch.load(checkpoint)['model_state_dict']
        model.load_state_dict({k.replace('quantizer.model','quantizer.vq'):v for k,v in pre_dic.items()})
        model.eval()
        return model
    
    @staticmethod
    def encodec_model_bw(checkpoint: str, bandwidth: float):
        """Return target bw model, if you train a model in a single bandwidth
        """
        import os
        assert os.path.exists(checkpoint), "checkpoint not exists"
        print("loading model from: ",checkpoint)
        target_bandwidths = bandwidth
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
                target_bandwidths, sample_rate, channels,
                causal=False, model_norm='time_group_norm', audio_normalize=True,
                segment=1., name='my_encodec')
        pre_dic = torch.load(checkpoint)['model_state_dict']
        model.load_state_dict({k.replace('quantizer.model','quantizer.vq'):v for k,v in pre_dic.items()})
        model.eval()
        return model


def test_encoder():
    encoder = EncodecEncoder()
    x = torch.randn(1, 1, 24000)
# The above code snippet is calling a function `encoder` with input `x`, and then printing the shape
# of the output `z`. It also includes an assertion to check if the shape of `z` is equal to [1, 128,
# 75]. If the assertion fails, it will raise an AssertionError with the actual shape of `z`.
    z = encoder(x)
    print(z.shape)
    assert list(z.shape) == [1, 128, 75], z.shape
    
def test_decoder():
    decoder = EncodecDecoder()
    x = torch.randn(1, 1, 24000)
    z = torch.randn(1, 128, 75)
    print(z.shape)
    y = decoder(z)
    assert y.shape == y.shape, (x.shape, y.shape)
    
def test_enc_dec():
    encoder = EncodecEncoder()
    decoder = EncodecDecoder()
    
    x = torch.randn(1, 1, 24000)
    print("input shape", x.shape)
    z = encoder(x)
    print("Encoder done ................ ")
    print("Encoder output shape", z.shape)
    assert list(z.shape) == [1, 128, 75], z.shape
    y = decoder(z)
    print("Decoder output shape", y.shape)
    assert y.shape == x.shape, (x.shape, y.shape)      
        
if __name__ == '__main__':
    test_enc_dec()