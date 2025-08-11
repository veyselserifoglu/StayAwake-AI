# StayAwake-AI
An AI-powered application designed to detect driver drowsiness in real-time, enhancing road safety by alerting drivers to potential fatigue.

## Project Structure

```
StayAwake-AI/
├── .devcontainer/                    # Development container configuration
├── diagrams/                         # Project diagrams and visualizations
├── docs/                            # Documentation files
│   ├── efficientnet_architecture_comparison.md
│   └── relevant data sets and resources.md
├── models_inattention/              # Model results and performance metrics
│   ├── B0/                         # EfficientNet-B0 results
│   ├── B1/                         # EfficientNet-B1 results
│   ├── B2/                         # EfficientNet-B2 results
│   ├── B4/                         # EfficientNet-B4 results
│   └── ImageNet_Pretrained_B4/     # Pre-trained B4 results
├── notebooks/                       # Jupyter notebooks
│   ├── driver-drowsiness-detection-case-study.ipynb
│   └── README.md
├── only_cm/                        # Confusion matrices and evaluation plots
├── only_models/                    # Trained model weights (.keras files)
├── scripts/                        # Utility scripts
│   ├── fetch_learning_rates.py
│   └── models_diagrams.py
├── model_card.md                   # Comprehensive model documentation
├── run notebook.md                 # Instructions for running notebooks
├── requirements.txt                # Python dependencies
└── README.md                      # Project documentation
```

## How to Run the Project

### 🚀 Recommended: Run on Kaggle (Easiest)

#### Option 1: View and Run Directly
1. **Visit the live notebook**: [Driver Drowsiness Detection on Kaggle](https://www.kaggle.com/code/fissalalsharef/driver-drowsiness-detection)
2. **View the results**: See all outputs, visualizations, and model performance
3. **Run immediately**: Click "Copy and Edit" to create your own editable version

#### Option 2: Start from Template
For the best experience with pre-configured environment and access to all datasets and models:

1. **Visit the Kaggle Notebook**: Access the complete notebook with all resources
2. **Click "Copy and Edit"**: Create your own version of the notebook
3. **Enable GPU** (recommended): Go to Settings → Accelerator → GPU
4. **Run All Cells**: Execute the entire notebook with one click

**What's included on Kaggle:**
- ✅ Complete driver inattention dataset
- ✅ Pre-trained EfficientNet models (B0, B1, B4)
- ✅ All required Python libraries
- ✅ Sample images for testing
- ✅ Visualization and evaluation code

### 🛠️ Alternative: Local Development

#### Prerequisites
- Python 3.8+
- GPU with CUDA support (recommended)
- Git

#### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/veyselserifoglu/StayAwake-AI.git
cd StayAwake-AI
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download required resources:**
   - [Trained model weights](https://www.kaggle.com/models/fissalalsharef/efficientnet_inattention_driver/)
   - [Training dataset](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset)
   - [OOD dataset](https://www.kaggle.com/datasets/amreen8441/annotated-driver-drowsiness)

4. **Adjust file paths** in the notebook to match your local directory structure

5. **Run the notebook:**
```bash
jupyter notebook notebooks/driver-drowsiness-detection-case-study.ipynb
```

**Note:** Local execution requires significant setup and may encounter path/dependency issues. Kaggle is strongly recommended for the best experience.

## Development Setup (Advanced)

You can choose between two development environments:

### Option 1: Local Development with VS Code Dev Container

#### Prerequisites
- Docker
- VS Code
- VS Code Remote - Containers extension

#### Getting Started
1. Clone the repository:
```bash
git clone https://github.com/veyselserifoglu/StayAwake-AI.git
cd StayAwake-AI
```

2. Open the project in VS Code:
```bash
code .
```

3. When prompted, click "Reopen in Container" to start the development environment.

4. The container will automatically:
   - Install all required dependencies
   - Set up the Python environment
   - Configure VS Code with recommended extensions

### Option 2: Direct Local Setup

1. Follow the local development setup instructions above
2. Use your preferred IDE or text editor
3. Install Jupyter for notebook development:
```bash
pip install jupyter
jupyter notebook
```

## Model Information

This project implements driver inattention detection using EfficientNet architectures:

- **EfficientNet-B0**: Lightweight model with 84.2% validation accuracy
- **EfficientNet-B1**: Balanced model with 86.1% validation accuracy  
- **EfficientNet-B4**: High-performance model with 88.7% validation accuracy
- **EfficientNet-B4 (ImageNet Pretrained)**: Best performing model with 91.3% validation accuracy

For detailed model information, see [`model_card.md`](model_card.md).

## Files Description

- **`notebooks/`**: Contains the main Jupyter notebook with complete implementation
- **`model_card.md`**: Comprehensive documentation of all models and their performance
- **`run notebook.md`**: Quick start guide for running the notebook
- **`models_inattention/`**: Performance metrics and results for each model variant
- **`only_models/`**: Pre-trained model weights in Keras format
- **`only_cm/`**: Confusion matrices and evaluation visualizations
- **`scripts/`**: Utility scripts for model analysis and visualization

## Contributing

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   - Branch naming convention: `feature/` for new features, `fix/` for bug fixes
   - Each branch should focus on a single feature or fix
   - Branches should not live longer than one week
   - If work takes longer than a week, create a new branch and reference the previous work

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

3. Push your branch and submit a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```
   - Create a pull request from your branch to `main`
   - Include a clear description of your changes
   - Reference any related issues

4. After your pull request is merged:
   - Delete your feature branch
   - Update your local `main` branch
   ```bash
   git checkout main
   git pull origin main
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
