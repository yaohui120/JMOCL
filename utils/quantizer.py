# changed from https://github.com/vineeths96/Compressed-Transformers
import torch
from torch import nn
from torch.nn import functional as F
import pdb

def change_type(tensor, bits):
    if bits == 32:
        tensor = tensor.to(dtype=torch.int32)
    elif bits == 16:
        tensor = tensor.to(dtype=torch.int16)
    elif bits == 8:
        tensor = tensor.to(dtype=torch.int8)
    return tensor

def quantize(tensor, bits):
    """
    Quantization function
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :return: Quantized code
    """
    # s = (1 << bits) - 1

    # # norm = torch.norm(tensor)
    # norm = tensor.abs().max()

    # sign_array = torch.sign(tensor)#.to(dtype=torch.int8)
    # sign_array = change_type(sign_array, bits)

    # l_array = torch.abs(tensor) / norm * s
    # # l_array_floored = l_array.to(dtype=torch.int8)
    # l_array_floored = change_type(l_array, bits)
    # prob_array = l_array - l_array_floored
    # prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

    # mask = torch.bernoulli(prob_array)
    # xi_array = l_array_floored + mask
    # xi_array = xi_array.to(dtype=torch.int32)

    # # sign_xi_array = (sign_array * xi_array).to(dtype=torch.int8)
    # sign_xi_array = change_type((sign_array * xi_array), bits)
    # norm = norm / s
    
    # pdb.set_trace()
    psize = (1 << bits) - 1
    # 不对称量化，0-psize
    # scale = (tensor.max()-tensor.min())/psize
    # zero_point = torch.round(-tensor.min()/scale).to(dtype=torch.int16)
    # x_q = (torch.round(tensor/scale)+zero_point)
    # x_q1 = torch.clamp(x_q, min=0, max=psize)
    # x_dq = (x_q1-zero_point)*scale
    # (tensor-x_dq).abs().sum()
    # 对称量化
    scale = max(tensor.max().abs(), tensor.min().abs())/psize
    x_q = torch.round(tensor/scale)
    x_q1 = torch.clamp(x_q, min=-psize/2, max=psize/2-1)
    x_q1 = change_type(x_q1, bits)
    # x_dq = x_q1 * scale
    # pdb.set_trace()
    # print((tensor-sign_xi_array*norm).abs().sum())
    # return norm, sign_xi_array
    return scale, x_q1 # == norm, sign_xi_array


def dequantize(norm, sign_xi_array):
    """
    Dequantize the quantization code
    :param norm: Norm of code
    :param sign_xi_array: Rounded vector of code
    :return: Dequantized weights
    """

    weights = norm * sign_xi_array

    return weights


class FakeLinearQuantizationFunction(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""

    @staticmethod
    def forward(ctx, input, bits=7):
        norm, quantized_weight = quantize(input, bits)
        return dequantize(norm, quantized_weight)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


_fake_quantize = FakeLinearQuantizationFunction.apply


class QuantizedLinear(nn.Linear):
    """Linear layer with quantization aware training capability"""

    def __init__(self, *args, weight_bits=16, warmup_step=0, **kwargs):
        super().__init__(*args, **kwargs)

        if weight_bits < 1:
            raise ValueError(f"weight_bits={weight_bits} must be higher than 0 ")

        self.weight_bits = weight_bits
        self.warmup_step = warmup_step
        
        self._fake_quantized_weight = None

        self._step = 0
        self.quantized_weight = None
        self.weight_norm = None
        
        # self.accumulation_bits = 32
        # self.quantized_bias = None
        # self.bias_norm = None
        
    def training_quantized_forward(self, input):
        """Fake quantizes weights. Function should only be used while training"""
        assert self.training, "Should be called only during training"

        self._fake_quantized_weight = _fake_quantize(self.weight, self.weight_bits)
        out = F.linear(input, self._fake_quantized_weight, self.bias)

        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. Function should be called only during inference"""
        assert not self.training, "Should be called only during inference"

        weight = self.weight_norm * self.quantized_weight

        # if self.bias is not None:
        #     bias = self.bias_norm * self.quantized_bias

        out = F.linear(input, weight, self.bias)

        return out

    def _eval(self):
        """Sets the model for inference by quantizing the model"""
        self.weight_norm, self.quantized_weight = quantize(self.weight, self.weight_bits)

        # if self.bias is not None:
        #     self.bias_norm, self.quantized_bias = quantize(self.bias, self.accumulation_bits)

    def forward(self, input):
        """Passes the input through the model during training and inference"""
        if self.training:
            # if self._step > self.warmup_step:
            #     out = self.training_quantized_forward(input)
            # else:
            #     out = super().forward(input)
            # self._step += 1
            out = self.training_quantized_forward(input)
            # out = F.linear(input, self.weight, self.bias)
        else:
            self._eval()
            out = self.inference_quantized_forward(input)
        # # pdb.set_trace()
        # out = F.linear(input, self.weight, self.bias)
        return out