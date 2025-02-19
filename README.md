# Machine Learning Project

This repository serves as a working space for me (Davy) and my friend Shiloh. We are collaborating on various machine learning tasks and experiments. The goal of this project is to explore different machine learning algorithms, models, and techniques to solve real-world problems.

## Table of Contents
- [Project Overview](#project-overview)
- [Recommended Environment setup](#Environment-Setup)

## Project Overview
This repository includes:
- Python scripts for data preprocessing
- Machine learning related practice projects
- Reference for financial context
- Datasets used for training and testing
- Documentation and notes on the steps and methodology


## Environment Setup

To get started with this project, it's recommended to set up a virtual environment (venv) to isolate dependencies and avoid conflicts with other projects. Below are the steps to set up your environment and install the required libraries.

### Step 1: Set up a Virtual Environment

1. First, navigate to the directory where your project is located.
2. Create a virtual environment by running the following command:

```bash
python -m venv venv
``` 

3. Activate the virtual environment:
   - On **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - On **MacOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

### Step 2: Install the Required Libraries

Once the virtual environment is activated, you can install the necessary packages for this project by running:

```bash
pip install numpy openpyxl pandas pyarrow pyjanitor ipykernel
```

These libraries are essential for various machine learning tasks in this project:
- **numpy**: For efficient numerical operations, especially matrix manipulation.
- **openpyxl**: For reading and writing Excel files, which might be useful for dataset manipulation.
- **pandas**: For data manipulation and analysis, including DataFrame structures.
- **pyarrow**: For handling large datasets and supporting Apache Arrow format.
- **pyjanitor**: For cleaning and preprocessing data in an easy-to-use manner.
- **ipykernel**: Required to run Jupyter notebooks and execute Python code in notebook format.

### Step 3: Verify Installation

To check if all the required libraries are installed correctly, run the following in your Python environment:

```python
import numpy as np
import openpyxl
import pandas as pd
import pyarrow
import janitor
import ipykernel
```

If no errors are raised, you're good to go!

### Step 4: Optional - Jupyter Notebook Setup

If you'd like to use Jupyter Notebooks for this project, you can install Jupyter by running:

```bash
pip install jupyterlab
```

Then, start a Jupyter notebook by running:

```bash
jupyter notebook
```

This will open the Jupyter dashboard in your web browser, where you can create and work with notebooks.
