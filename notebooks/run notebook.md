# How to Run the Driver Inattention Detection Notebook

## Important Notice

**This notebook should be run on Kaggle platform for optimal performance and access to all required resources.**

## Why Kaggle?

- **Pre-configured Environment**: All required datasets and trained models are already available on Kaggle
- **No Setup Required**: No need to download large model files or datasets locally
- **GPU Access**: Free GPU acceleration available for inference and visualization
- **One-Click Execution**: Everything is ready to run without additional configuration

## Getting Started

> **⚠️ IMPORTANT DISCLAIMER ⚠️**  
> **Before running any cells**: First execute ONLY the library installation cell (usually the first code cell), then **RESTART THE KERNEL** before running any other cells. This ensures all dependencies are properly installed and prevents runtime errors.

### Option 1: View and Run Directly
1. **Visit the live notebook**: [Driver Drowsiness Detection on Kaggle](https://www.kaggle.com/code/fissalalsharef/driver-drowsiness-detection)
2. **View the results**: See all outputs, visualizations, and model performance
3. **Run immediately**: Click "Copy and Edit" to create your own editable version

### Option 2: Start from Scratch
1. **Visit the Kaggle Notebook**: Access the notebook template
2. **Click "Copy and Edit"**: This creates your own version of the notebook
3. **Enable GPU** (recommended): Go to Settings → Accelerator → GPU
4. **Run All Cells**: After following the disclaimer above, use Ctrl+A to select all cells, then Shift+Enter to run

## What's Included on Kaggle

- ✅ Complete driver inattention dataset
- ✅ Pre-trained EfficientNet models (B0, B1, B4)
- ✅ All required Python libraries
- ✅ Sample images for testing
- ✅ Visualization and evaluation code

## Local Execution (Not Recommended)

If you prefer to run locally, you'll need to:
- Fork or download the [GitHub repository](https://github.com/veyselserifoglu/StayAwake-AI)
- Download the [trained model weights](https://www.kaggle.com/models/fissalalsharef/efficientnet_inattention_driver/)
- Download the [training dataset](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset)
- Download the [OOD dataset](https://www.kaggle.com/datasets/amreen8441/annotated-driver-drowsiness)
- Install all dependencies from `requirements.txt`
- Adjust file paths in the notebook

**Note**: Local execution requires significant setup and may encounter path/dependency issues.

## Questions or Issues?

If you encounter any problems running the notebook on Kaggle, please check:
1. GPU is enabled in notebook settings
2. All cells are executed in order
3. Kaggle dataset is properly linked

---

**For the best experience and to avoid setup complications, please use Kaggle!**
