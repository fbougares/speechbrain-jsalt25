"""

The goal of this code is to integrate "EnCodec: High Fidelity Neural Audio Compression"  from Meta inside SpeechBrain



Authors
 * Fethi Bougares, 2025

"""


import torch
import logging

import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from collections import defaultdict
from librosa.filters import mel as librosa_mel_fn
import typing as tp


logging.getLogger().setLevel(logging.INFO)

##### Audio to Mel ##### 
class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        device='cuda'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(sr=sampling_rate,n_fft=n_fft,n_mels=n_mel_channels,fmin=mel_fmin,fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).cuda().float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audioin):
        # turn each channel into a batch
        shape = audioin.shape
        if len(shape) > 2:
            audioin = audioin.reshape(shape[0] * shape[1], -1)
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect")
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False,
        )
        mel_output = torch.matmul(self.mel_basis, torch.sum(torch.pow(fft, 2), dim=[-1]))
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        # restore original shape [B, H, T]
        if len(shape) > 2:
            log_mel_spec = log_mel_spec.reshape(shape[0], shape[1], -1)
        return log_mel_spec


##### Torch distributed utils ########

def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_distributed():
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _check_number_of_params(params: tp.List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                            "at least one worker has a different one.")


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def sync_buffer(buffers, average=True):
    """
    Sync grad for buffers. If average is False, broadcast instead of averaging.
    """
    if not is_distributed():
        return
    handles = []
    for buffer in buffers:
        if torch.is_floating_point(buffer.data):
            if average:
                handle = torch.distributed.all_reduce(
                    buffer.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            else:
                handle = torch.distributed.broadcast(
                    buffer.data, src=0, async_op=True)
            handles.append((buffer, handle))
    for buffer, handle in handles:
        handle.wait()
        if average:
            buffer.data /= world_size


def sync_grad(params):
    """
    Simpler alternative to DistributedDataParallel, that doesn't rely
    on any black magic. For simple models it can also be as fast.
    Just call this on your model parameters after the call to backward!
    """
    if not is_distributed():
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handle = torch.distributed.all_reduce(
                p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            handles.append((p, handle))
    for p, handle in handles:
        handle.wait()
        p.grad.data /= world_size()


def average_metrics(metrics: tp.Dict[str, float], count=1.):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))


#### Total loss calculation ############@

def total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=24000):
    """This function is used to compute the total loss of the encodec generator.
        Loss = \lambda_t * L_t + \lambda_f * L_f + \lambda_g * L_g + \lambda_feat * L_feat
        L_t: time domain loss | L_f: frequency domain loss | L_g: generator loss | L_feat: feature loss
        \lambda_t = 0.1       | \lambda_f = 1              | \lambda_g = 3       | \lambda_feat = 3
    Args:
        fmap_real (list): fmap_real is the output of the discriminator when the input is the real audio. 
            len(fmap_real) = len(fmap_fake) = disc.num_discriminators = 3
        logits_fake (_type_): logits_fake is the list of every sub discriminator output of the Multi discriminator 
            logits_fake, _ = disc_model(model(input_wav)[0].detach())
        fmap_fake (_type_): fmap_fake is the output of the discriminator when the input is the fake audio.
            fmap_fake = disc_model(model(input_wav)[0]) = disc_model(reconstructed_audio)
        input_wav (tensor): input_wav is the input audio of the generator (GT audio)
        output_wav (tensor): output_wav is the output of the generator (output = model(input_wav)[0])
        sample_rate (int, optional): Defaults to 24000.

    Returns:
        loss: total loss
    """
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')
    # Collect losses as defined in paper for use with balancer
    # l_t - L1 distance between the target and compressed audio over the time domain
    # l_f - linear combination between the L1 and L2 losses over the mel-spectrogram using several time scales
    # l_g - adversarial loss for the generator
    # l_feat - relative feature matching loss for the generator
    l_t = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_f = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_g = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_feat = torch.tensor([0.0], device='cuda', requires_grad=True)

    #time domain loss, output_wav is the output of the generator
    l_t = l1Loss(input_wav, output_wav) 

    #frequency domain loss, window length is 2^i, hop length is 2^i/4, i \in [5,11]. combine l1 and l2 loss
    for i in range(5, 12): #e=5,...,11
        fft = Audio2Mel(n_fft=2 ** i,win_length=2 ** i, hop_length=(2 ** i) // 4, n_mel_channels=64, sampling_rate=sample_rate)
        l_f = l_f + l1Loss(fft(input_wav), fft(output_wav)) + l2Loss(fft(input_wav), fft(output_wav))

    #generator loss and feat loss, D_k(\hat x) = logits_fake[k], D_k^l(x) = fmap_real[k][l], D_k^l(\hat x) = fmap_fake[k][l]
    # l_g = \sum max(0, 1 - D_k(\hat x)) / K, K = disc.num_discriminators = len(fmap_real) = len(fmap_fake) = len(logits_fake) = 3
    # l_feat = \sum |D_k^l(x) - D_k^l(\hat x)| / |D_k^l(x)| / KL, KL = len(fmap_real[0])*len(fmap_real)=3 * 5
    for tt1 in range(len(fmap_real)): # len(fmap_real) = 3
        l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
        for tt2 in range(len(fmap_real[tt1])): # len(fmap_real[tt1]) = 5
            # l_feat = l_feat + l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2].detach()))
            l_feat = l_feat + l1Loss(fmap_real[tt1][tt2], fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))

    KL_scale = len(fmap_real)*len(fmap_real[0]) # len(fmap_real) == len(fmap_fake) == len(logits_real) == len(logits_fake) == disc.num_discriminators == K
    l_feat /= KL_scale
    K_scale = len(fmap_real) # len(fmap_real[0]) = len(fmap_fake[0]) == L
    l_g /= K_scale

    return {
        'l_t': l_t,
        'l_f': l_f,
        'l_g': l_g,
        'l_feat': l_feat,
    }




### discriminator loss ### 
def disc_loss(logits_real, logits_fake):
    """This function is used to compute the loss of the discriminator.
        l_d = \sum max(0, 1 - D_k(x)) + max(0, 1 + D_k(\hat x)) / K, K = disc.num_discriminators = len(logits_real) = len(logits_fake) = 3
    Args:
        logits_real (List[torch.Tensor]): logits_real = disc_model(input_wav)[0]
        logits_fake (List[torch.Tensor]): logits_fake = disc_model(model(input_wav)[0])[0]

    Returns:
        lossd: discriminator loss
    """
    relu = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd

###         Loss Balancer #####

def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: tp.Dict[str, tp.Any], weight: float = 1) -> tp.Dict[str, float]:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


class Balancer:
    """Loss balancer.

    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.

    Expected usage:
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)

    ..Warning:: It is unclear how this will interact with DistributedDataParallel,
        in particular if you have some losses not handled by the balancer. In that case
        you can use `encodec.distrib.sync_grad(model.parameters())` and
        `encodec.distrib.sync_buffwers(model.buffers())` as a safe alternative.

    Args:
        weights (Dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        rescale_grads (bool): Whether to rescale gradients or not, without. If False, this is just
            a regular weighted sum of losses.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        emay_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): Whether to store additional ratio for each loss key in metrics.
    """

    def __init__(self, weights: tp.Dict[str, float], rescale_grads: bool = True, total_norm: float = 1.,
                 ema_decay: float = 0.999, per_batch_item: bool = True, epsilon: float = 1e-12,
                 monitor: bool = False):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

    @property
    def metrics(self):
        return self._metrics

    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor, retain_graph=False):
        norms = {}
        grads = {}
        for name, loss in losses.items():
            grad, = autograd.grad(loss, [input], retain_graph=True)
            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims).mean()
            else:
                norm = grad.norm()
            norms[name] = norm
            grads[name] = grad

        count = 1
        if self.per_batch_item:
            count = len(grad)
        avg_norms = average_metrics(self.averager(norms), count)
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
            for k, v in avg_norms.items():
                self._metrics[f'ratio_{k}'] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_grad: tp.Any = 0
        for name, avg_norm in avg_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                grad = grads[name] * scale
            else:
                grad = self.weights[name] * grads[name]
            out_grad += grad
        input.backward(out_grad, retain_graph=retain_graph)
        

def test():
    from torch.nn import functional as F
    x = torch.zeros(1, requires_grad=True)
    one = torch.ones_like(x)
    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}

    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=False)
    balancer.backward(losses, x)
    assert torch.allclose(x.grad, torch.tensor(99.)), x.grad

    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}
    x.grad = None
    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=True)
    balancer.backward({'1': loss_1, '2': loss_2}, x)
    assert torch.allclose(x.grad, torch.tensor(0.)), x.grad


if __name__ == '__main__':
    test()