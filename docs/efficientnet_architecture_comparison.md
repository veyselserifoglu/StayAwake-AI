# EfficientNet Architecture Comparison: B0 through B4

## Introduction

The EfficientNet family represents a systematic approach to model scaling, using a compound coefficient to uniformly scale network width, depth, and resolution. This document compares the key architectural differences between EfficientNet B0, B1, B2, B3, and B4 variants.

## Scaling Coefficients and Model Sizes

The EfficientNet variants differ in three key dimensions that are scaled according to specific coefficients:

| Model Version | Depth Coefficient | Width Coefficient | Resolution | Parameters | Input Size |
|---------------|-------------------|-------------------|------------|------------|------------|
| B0 (baseline) | 1.0               | 1.0               | 224×224    | 5.3M       | 224×224    |
| B1            | 1.1               | 1.0               | 240×240    | 7.8M       | 240×240    |
| B2            | 1.2               | 1.1               | 260×260    | 9.2M       | 260×260    |
| B3            | 1.4               | 1.2               | 300×300    | 12M        | 300×300    |
| B4            | 1.8               | 1.4               | 380×380    | 19M        | 380×380    |

## Understanding Scaling Dimensions

### Depth Coefficient
- Controls the number of layers in the network
- A depth coefficient of 1.8 (B4) means approximately 1.8× more layers than the baseline (B0)
- This increases the network's capacity to model more complex features and hierarchical relationships

### Width Coefficient
- Controls the number of channels in each layer
- A width coefficient of 1.4 (B4) means approximately 1.4× more channels than the baseline (B0)
- More channels allow the network to capture more features at each level of abstraction

### Resolution
- Determines the input image size and affects feature map dimensions throughout the network
- Higher resolutions preserve more fine details but require more computation
- The resolution scaling from 224×224 (B0) to 380×380 (B4) significantly increases the computational cost

## Layer Counts and Network Structure

### Total Layers
- **B0**: 237 total layers
- **B1**: 277 total layers
- **B2**: 304 total layers
- **B3**: 362 total layers
- **B4**: 471 total layers

### MBConv Block Distribution
The number of Mobile Inverted Bottleneck ConvBlock (MBConv) blocks in each stage increases with model size:

| Stage | Block Type | B0 | B1 | B2 | B3 | B4 |
|-------|------------|----|----|----|----|-----|
| 1     | MBConv1    | 1  | 2  | 2  | 2  | 2   |
| 2     | MBConv6    | 2  | 3  | 3  | 3  | 4   |
| 3     | MBConv6    | 2  | 3  | 3  | 4  | 4   |
| 4     | MBConv6    | 3  | 4  | 4  | 5  | 6   |
| 5     | MBConv6    | 3  | 4  | 4  | 5  | 6   |
| 6     | MBConv6    | 4  | 5  | 5  | 6  | 8   |
| 7     | MBConv6    | 1  | 2  | 2  | 2  | 2   |

### Channel Widths
The number of channels in each stage also increases with model size:

| Stage | B0    | B1    | B2    | B3    | B4    |
|-------|-------|-------|-------|-------|-------|
| 1     | 32    | 32    | 32    | 40    | 48    |
| 2     | 16    | 16    | 16    | 24    | 24    |
| 3     | 24    | 24    | 24    | 32    | 32    |
| 4     | 40    | 40    | 48    | 48    | 56    |
| 5     | 80    | 80    | 88    | 96    | 112   |
| 6     | 112   | 112   | 120   | 136   | 160   |
| 7     | 192   | 192   | 208   | 232   | 272   |
| 8     | 320   | 320   | 352   | 384   | 448   |

## Implications for Training from Scratch

The substantial differences in parameter counts and architecture complexity have significant implications for training from scratch:

1. **Data Requirements**: Larger models like B4 (19M parameters) require significantly more training data than B0 (5.3M parameters) to avoid overfitting
   
2. **Optimization Difficulty**: Deeper networks are harder to optimize from random initialization due to vanishing/exploding gradient problems

3. **Computational Cost**: Training larger models requires more memory and computation time, which may not be justified if the dataset size is limited

4. **Generalization**: On smaller datasets, smaller models often generalize better despite having less theoretical capacity

5. **Parameter Efficiency**: Smaller models like B0 and B1 typically achieve higher accuracy per parameter when training data is limited

This architectural comparison helps explain why EfficientNet B1 might outperform B4 when training from scratch on limited datasets, despite B4's greater theoretical capacity.
