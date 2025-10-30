# Code Quality Assessment Tools

A comprehensive suite of automated code quality checking tools for Python, supporting multiple code analysis tools (Pylint, SonarQube, Radon, Bandit).

## Project Overview

This project provides four independent code checkers that extract code from JSON-formatted submission files and perform quality assessments. Each checker invokes its corresponding code analysis tool to generate detailed evaluation reports.

## Project Structure

```
.
├── submission_code/          # Code submission files storage directory
├── submissions/              # JSON-formatted submission data directory
├── pylint_checker.py         # Pylint code quality checker
├── sonarqube_checker.py      # SonarQube code quality checker
├── radon_checker.py          # Radon code complexity checker
├── bandit_checker.py         # Bandit security vulnerability checker
├── pylint_summary_generator.py      # Pylint result summary generator (optional)
├── sonarqube_summary_generator.py   # SonarQube result summary generator (optional)
├── radon_summary_generator.py       # Radon result summary generator (optional)
├── bandit_summary_generator.py      # Bandit result summary generator (optional)
├── requirements.txt          # Python dependency list
└── README.md                 # Project documentation
```

## Requirements

- **Python**: 3.12+
- **Operating System**: Linux / macOS / Windows
- **Required Tools**: 
  - Pylint
  - Radon
  - Bandit
  - SonarQube Scanner (for SonarQube checks)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JueH-96/Code-Quality-Evaluation.git
cd Code-Quality-Evaluation
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Code Analysis Tools

```bash
# Install Pylint
pip install pylint

# Install Radon
pip install radon

# Install Bandit
pip install bandit

```

## Usage

### Running Checkers

Each checker can be executed in two ways:

**Method 1: Direct Python Execution**

```bash
python3 pylint_checker.py
python3 sonarqube_checker.py
python3 radon_checker.py
python3 bandit_checker.py
```

**Method 2: Execute After Granting Permissions**

```bash
chmod +x pylint_checker.py
./pylint_checker.py

chmod +x sonarqube_checker.py
./sonarqube_checker.py

chmod +x radon_checker.py
./radon_checker.py

chmod +x bandit_checker.py
./bandit_checker.py
```

### Quick Summary Generation (From Raw Data)

If raw data already exists, you can directly run summary generators to quickly obtain results:

```bash
python3 pylint_summary_generator.py
python3 sonarqube_summary_generator.py
python3 radon_summary_generator.py
python3 bandit_summary_generator.py
```

## Output Results

Results from each checker are saved in corresponding result directories:

### Pylint Result Directory (`pylint_result/`)

```
pylint_result/
├── pylint_summary.csv              # Summary results file
├── LLM_Difficulty_Pylint.csv       # Detailed evaluation results grouped by LLM
pylint_raw/                     # Raw data directory
    ├── <llm_name>/
    │   ├── <question_id>_raw.txt
    │   └── ...
    └── ...
```

### SonarQube Result Directory (`sonarqube_result/`)

```
sonarqube_result/
├── sonarqube_summary.csv           # Summary results file
├── LLM_Difficulty_SonarQube.csv    # Detailed evaluation results grouped by LLM
sonarqube_raw/                  # Raw data directory
    ├── <llm_name>/
    │   ├── <llm_name>_page1.txt
    │   └── ...
    └── ...
```

### Radon Result Directory (`radon_result/`)

```
radon_result/
├── radon_summary.csv               # Summary results file
├── LLM_Difficulty_Radon.csv        # Detailed evaluation results grouped by LLM
radon_raw/                      # Raw data directory
    ├── <llm_name>/
    │   ├── <question_id>_cc.txt
    │   ├── <question_id>_mi.txt
    │   └── ...
    └── ...
```

### Bandit Result Directory (`bandit_result/`)

```
bandit_result/
├── bandit_summary.csv              # Summary results file
├── LLM_Difficulty_Bandit.csv       # Detailed evaluation results grouped by LLM
bandit_raw/                     # Raw data directory
    ├── <llm_name>/
    │   ├── <question_id>_raw.json
    │   └── ...
    └── ...
```

## Result Files Description

### 1. XXX_summary.csv (Summary Results)

Contains overall evaluation metric summaries for all LLMs, such as:
- Average code quality scores
- Issue count statistics
- Pass rates, etc.

### 2. LLM_Difficulty_XXX.csv (Detailed Results)

Records detailed evaluation results for each code sample by LLM, including:
- Question ID
- Difficulty level
- Specific metrics (e.g., Pylint scores, cyclomatic complexity, security issue counts, etc.)

### 3. XXX_raw/ (Raw Data)

Stores raw output data from each check, useful for:
- Troubleshooting
- Data reproduction
- Custom analysis

## Checker Functionality

### Pylint Checker
- **Function**: Comprehensive code quality analysis
- **Output Metrics**: Code score (0-10), warning count, error count, code convention issues, etc.

### SonarQube Checker
- **Function**: Comprehensive code quality analysis
- **Output Metrics**: Bug count, code smells, security vulnerabilities, technical debt, etc.

### Radon Checker
- **Function**: Comprehensive code quality analysis
- **Output Metrics**: Cyclomatic Complexity (CC), Maintainability Index (MI), etc.

### Bandit Checker
- **Function**: Security vulnerability detection
- **Output Metrics**: High/medium/low risk issue counts, specific security vulnerability types, etc.

## Important Notes

1. **JSON Format Requirements**: Ensure JSON files in the `submissions/` directory are correctly formatted with necessary code fields
2. **Tool Installation**: Verify all code analysis tools are properly installed before running
3. **Permission Issues**: On Linux/macOS, ensure execution permissions are granted before running scripts
4. **SonarQube Configuration**: SonarQube checker requires configuration of SonarQube server connection information
5. **Resource Usage**: Batch checking large amounts of code may require significant time and system resources

## Troubleshooting

### Issue 1: Tool Not Installed Error

```bash
# Check if tools are installed
pylint --version
radon --version
bandit --version
```

### Issue 2: Permission Denied

```bash
# Grant execution permissions
chmod +x *.py
```

### Issue 3: JSON Files Not Found

Ensure the `submissions/` directory exists and contains properly formatted JSON files.
