import torch
import torch.nn as nn
#added logic of making division not zero
def quantize_tensor(tensor, scale, zero_point):
    
    q_tensor = (tensor / scale).round() + zero_point
    return q_tensor.clamp(0, 255).to(torch.uint8)

def dequantize_tensor(q_tensor, scale, zero_point):
    return (q_tensor.to(torch.float32) - zero_point) * scale

def calculate_dynamic_scale_and_zero_point(tensor):
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / 255.0
   
    if scale == 0:
        scale = 1e-8
    zero_point = (-min_val / scale).round()
    zero_point = zero_point.clamp(0, 255)
    return scale, zero_point

def quantize_model(model):
    quantized_weights = {}
    scales = {}
    zero_points = {}

    for name, param in model.named_parameters():

        if ('weight' in name or "bias" in name) and len(param.shape)>1:
            print(name)
            scale, zero_point = calculate_dynamic_scale_and_zero_point(param.data)
            quantized_weights[name] = quantize_tensor(param.data, scale, zero_point)
            scales[name] = scale
            zero_points[name] = zero_point
        else:
            quantized_weights[name] = param.data
            
    for name, param in model.named_buffers():
        if 'Weight' in name:
            print(name)
            scale, zero_point = calculate_dynamic_scale_and_zero_point(param.data)
            quantized_weights[name] = quantize_tensor(param.data, scale, zero_point)
            scales[name] = scale
            zero_points[name] = zero_point
        else:
            quantized_weights[name] = param.data
    
    return quantized_weights, scales, zero_points

def apply_quantized_weights(model, quantized_weights, scales, zero_points):
    for name, param in model.named_parameters():
        if ('weight' in name or "bias" in name) and len(param.shape)>1:
            
            scale = scales[name]
            zero_point = zero_points[name]
            param.data = dequantize_tensor(quantized_weights[name], scale, zero_point)
        else:
            param.data = quantized_weights[name]
           
    for name, param in model.named_buffers():
        if 'Weight' in name:
           
            scale = scales[name]
            zero_point = zero_points[name]
            param.data = dequantize_tensor(quantized_weights[name], scale, zero_point)
        else:
           
            param.data = quantized_weights[name]
    return model
def save_quantized_model(quantized_weights, scales, zero_points, path):
    
    print(scales.keys())
    torch.save({'weights': quantized_weights, 'scales': scales, 'zero_points': zero_points}, path)

def load_quantized_model(model, path):
    
    checkpoint = torch.load(path)
    quantized_weights = checkpoint['weights']
    scales = checkpoint['scales']
    zero_points = checkpoint['zero_points']
    
    model = apply_quantized_weights(model, quantized_weights, scales, zero_points)
    return model
