# StayAwake-AI
An AI-powered application designed to detect driver drowsiness in real-time, enhancing road safety by alerting drivers to potential fatigue.

## Project Structure

```
StayAwake-AI/
├── .devcontainer/         # Development container configuration
├── data/                  # Data directory
│   ├── raw/              # Raw, immutable data
│   └── processed/        # Processed data
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── data/            # Data processing scripts
│   ├── models/          # Model definitions
│   ├── features/        # Feature engineering
│   └── visualization/   # Visualization tools
├── tests/               # Test files
├── models/              # Trained models
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Development Setup

You can choose between two development environments:

### Option 1: Local Development with VS Code Dev Container

#### Prerequisites

- Docker
- VS Code
- VS Code Remote - Containers extension

#### Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StayAwake-AI.git
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

### Option 2: Google Colab

#### Getting Started with Colab

1. Go to [Google Colab](https://colab.research.google.com/)

2. Choose one of these methods to connect to the repository:

   **Method 1: Using Colab's GitHub Integration**
   - Click on "File" > "Open notebook"
   - Select the "GitHub" tab
   - Enter the repository URL: `https://github.com/yourusername/StayAwake-AI`
   - Select the notebook you want to open

   **Method 2: Cloning the Repository**
   - Create a new notebook
   - Run the following commands in a code cell:
   ```python
   !git clone https://github.com/yourusername/StayAwake-AI.git
   %cd StayAwake-AI
   ```

3. To save your work back to GitHub, you have two options:

   **Option A: Using Git Commands**
   ```python
   !git add .
   !git commit -m "your commit message"
   !git push
   ```

   **Option B: Using Colab's Save to GitHub Feature**
   - Click on "File" > "Save a copy in GitHub"
   - Select your repository: `StayAwake-AI`
   - Choose the appropriate path:
     - For new notebooks: `notebooks/your_notebook_name.ipynb`
     - For modified notebooks: Keep the same path as the original file
   - Add a commit message
   - Click "OK"

4. To use GPU acceleration (recommended for training):
   - Click on "Runtime" > "Change runtime type"
   - Select "GPU" as the hardware accelerator
   - Click "Save"

### Using Jupyter Notebooks

#### Local Development
1. Start the Jupyter server:
```bash
jupyter notebook
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8888)

3. Create new notebooks in the `notebooks` directory or open existing ones

#### Google Colab
1. Follow the setup instructions in the "Google Colab" section above
2. Create new notebooks or open existing ones from the `notebooks` directory
3. Save your work using either of the GitHub save methods described above

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
