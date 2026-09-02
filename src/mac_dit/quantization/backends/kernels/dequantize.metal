#include <metal_stdlib>
using namespace metal;

kernel void dequantize_int8(
    const device char* qweight [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& in_features [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= numel) {
    return;
  }
  uint row = index / in_features;
  output[index] = half(qweight[index]) * scales[row];
}

kernel void dequantize_int4(
    const device uchar* qweight [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& padded_in_features [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= numel) {
    return;
  }

  uchar packed = qweight[index / 2];
  int quantized = (index & 1) == 0
      ? int(packed & 0x0F)
      : int(packed >> 4);
  quantized -= 8;

  uint row = index / padded_in_features;
  uint column = index - row * padded_in_features;
  uint groups_per_row = padded_in_features / group_size;
  uint group = column / group_size;
  half scale = scales[row * groups_per_row + group];
  output[index] = half(quantized) * scale;
}

// W8A16 Linear：权重保持 INT8，在乘法循环中按输出通道应用 scale。
kernel void linear_int8(
    const device half* input [[buffer(0)]],
    const device char* qweight [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    constant uint& numel [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= numel) {
    return;
  }

  uint row = index / out_features;
  uint output_column = index - row * out_features;
  uint input_offset = row * in_features;
  uint weight_offset = output_column * in_features;
  float accumulator = 0.0f;

  for (uint column = 0; column < in_features; ++column) {
    accumulator += float(input[input_offset + column])
        * float(qweight[weight_offset + column]);
  }

  accumulator *= float(scales[output_column]);
  if (has_bias != 0) {
    accumulator += float(bias[output_column]);
  }
  output[index] = half(accumulator);
}

// W4A16 Linear：每次从一个字节解包两个 INT4 权重，并按 group 应用 scale。
kernel void linear_int4(
    const device half* input [[buffer(0)]],
    const device uchar* qweight [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& padded_in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    constant uint& numel [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= numel) {
    return;
  }

  uint row = index / out_features;
  uint output_column = index - row * out_features;
  uint input_offset = row * in_features;
  uint weight_offset = output_column * padded_in_features;
  uint groups_per_row = padded_in_features / group_size;
  uint scale_offset = output_column * groups_per_row;
  float accumulator = 0.0f;

  for (uint column = 0; column < in_features; ++column) {
    uint weight_index = weight_offset + column;
    uchar packed = qweight[weight_index / 2];
    int quantized = (weight_index & 1) == 0
        ? int(packed & 0x0F)
        : int(packed >> 4);
    quantized -= 8;
    float scale = float(scales[scale_offset + column / group_size]);
    accumulator += float(input[input_offset + column])
        * float(quantized) * scale;
  }

  if (has_bias != 0) {
    accumulator += float(bias[output_column]);
  }
  output[index] = half(accumulator);
}
