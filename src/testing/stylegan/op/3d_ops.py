import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _setup_filter_kernel(filter_kernel, gain=1, up_factor=1, dim=2):
    """
    Set up a filter kernel and return it as a tensor.
    Arguments:
        filter_kernel (int, list, torch.tensor, None): The filter kernel
            values to use. If this value is an int, a binomial filter of
            this size is created. If a sequence with a single axis is used,
            it will be expanded to the number of `dims` specified. If value
            is None, a filter of values [1, 1] is used.
        gain (float): Gain of the filter kernel. Default value is 1.
        up_factor (int): Scale factor. Should only be given for upscaling filters.
            Default value is 1.
        dim (int): Number of dimensions of data. Default value is 2.
    Returns:
        filter_kernel_tensor (torch.Tensor)
    """
    filter_kernel = filter_kernel or 2
    if isinstance(filter_kernel, (int, float)):
        def binomial(n, k):
            if k in [1, n]:
                return 1
            return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))

        filter_kernel = [binomial(filter_kernel, k) for k in range(1, filter_kernel + 1)]
    if not torch.is_tensor(filter_kernel):
        filter_kernel = torch.tensor(filter_kernel)
    filter_kernel = filter_kernel.float()
    if filter_kernel.dim() == 1:
        _filter_kernel = filter_kernel.unsqueeze(0)
        while filter_kernel.dim() < dim:
            filter_kernel = torch.matmul(
                filter_kernel.unsqueeze(-1), _filter_kernel)
    assert all(filter_kernel.size(0) == s for s in filter_kernel.size())
    filter_kernel /= filter_kernel.sum()
    filter_kernel *= gain * up_factor ** 2
    return filter_kernel.float()


def _apply_conv(input, *args, transpose=False, **kwargs):
    """
    Perform a 1d, 2d or 3d convolution with specified
    positional and keyword arguments. Which type of
    convolution that is used depends on shape of data.
    Arguments:
        input (torch.Tensor): The input data for the
            convolution.
        *args: Positional arguments for the convolution.
    Keyword Arguments:
        transpose (bool): Transpose the convolution.
            Default value is False
        **kwargs: Keyword arguments for the convolution.
    """
    dim = input.dim() - 2
    conv_fn = getattr(
        F, 'conv{}{}d'.format('_transpose' if transpose else '', dim))
    return conv_fn(input=input, *args, **kwargs)


class FilterLayer(nn.Module):
    """
    Apply a filter by using convolution.
    Arguments:
        filter_kernel (torch.Tensor): The filter kernel to use.
            Should be of shape `dims * (k,)` where `k` is the
            kernel size and `dims` is the number of data dimensions
            (excluding batch and channel dimension).
        stride (int): The stride of the convolution.
        pad0 (int): Amount to pad start of each data dimension.
            Default value is 0.
        pad1 (int): Amount to pad end of each data dimension.
            Default value is 0.
        pad_mode (str): The padding mode. Default value is 'constant'.
        pad_constant (float): The constant value to pad with if
            `pad_mode='constant'`. Default value is 0.
    """

    def __init__(self,
                 filter_kernel,
                 stride=1,
                 pad0=0,
                 pad1=0,
                 pad_mode='constant',
                 pad_constant=0,
                 *args,
                 **kwargs):
        super(FilterLayer, self).__init__()
        dim = filter_kernel.dim()
        filter_kernel = filter_kernel.view(1, 1, *filter_kernel.size())
        self.register_buffer('filter_kernel', filter_kernel)
        self.stride = stride
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
            self.pad_mode = pad_mode
            self.pad_constant = pad_constant

    def forward(self, input, **kwargs):
        """
        Pad the input and run the filter over it
        before returning the new values.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        conv_kwargs = dict(
            weight=self.filter_kernel.repeat(
                input.size(1), *[1] * (self.filter_kernel.dim() - 1)),
            stride=self.stride,
            groups=input.size(1),
        )
        if self.fused_pad:
            conv_kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(
            input=x,
            transpose=False,
            **conv_kwargs
        )

    def extra_repr(self):
        return 'filter_size={}, stride={}'.format(
            tuple(self.filter_kernel.size()[2:]), self.stride)


class Upsample(nn.Module):
    """
    Performs upsampling without learnable parameters that doubles
    the size of data.
    Arguments:
        mode (str): 'FIR' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        gain (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
    """

    def __init__(self,
                 mode='FIR',
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 gain=1,
                 dim=2,
                 *args,
                 **kwargs):
        super(Upsample, self).__init__()
        assert mode != 'max', 'mode \'max\' can only be used for downsampling.'
        if mode == 'FIR':
            if filter is None:
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(
                filter_kernel=filter,
                gain=gain,
                up_factor=2,
                dim=dim
            )
            pad = filter_kernel.size(-1) - 1
            self.filter = FilterLayer(
                filter_kernel=filter_kernel,
                pad0=(pad + 1) // 2 + 1,
                pad1=pad // 2,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant
            )
            self.register_buffer('weight', torch.ones(*[1 for _ in range(dim + 2)]))
        self.mode = mode

    def forward(self, input, **kwargs):
        """
        Upsample inputs.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if self.mode == 'FIR':
            x = _apply_conv(
                input=input,
                weight=self.weight.expand(input.size(1), *self.weight.size()[1:]),
                groups=input.size(1),
                stride=2,
                transpose=True
            )
            x = self.filter(x)
        else:
            interp_kwargs = dict(scale_factor=2, mode=self.mode)
            if 'linear' in self.mode or 'cubic' in self.mode:
                interp_kwargs.update(align_corners=False)
            x = F.interpolate(input, **interp_kwargs)
        return x

    def extra_repr(self):
        return 'resample_mode={}'.format(self.mode)


class Downsample(nn.Module):
    """
    Performs downsampling without learnable parameters that
    reduces size of data by half.
    Arguments:
        mode (str): 'FIR', 'max' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        gain (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
    """

    def __init__(self,
                 mode='FIR',
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 gain=1,
                 dim=2,
                 *args,
                 **kwargs):
        super(Downsample, self).__init__()
        if mode == 'FIR':
            if filter is None:
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(
                filter_kernel=filter,
                gain=gain,
                up_factor=1,
                dim=dim
            )
            pad = filter_kernel.size(-1) - 2
            pad0 = pad // 2
            pad1 = pad - pad0
            self.filter = FilterLayer(
                filter_kernel=filter_kernel,
                stride=2,
                pad0=pad0,
                pad1=pad1,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant
            )
        self.mode = mode

    def forward(self, input, **kwargs):
        """
        Downsample inputs to half its size.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if self.mode == 'FIR':
            x = self.filter(input)
        elif self.mode == 'max':
            return getattr(F, 'max_pool{}d'.format(input.dim() - 2))(input)
        else:
            x = F.interpolate(input, scale_factor=0.5, mode=self.mode)
        return x

    def extra_repr(self):
        return 'resample_mode={}'.format(self.mode)
