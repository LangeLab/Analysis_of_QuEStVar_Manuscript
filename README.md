# QuEStVar Manuscript Analysis

| **`Status`** | **`License`** | **`Language`** | **`Release`** | **`DOI`** |  **`Citation`** |
|----------------|----------------|----------------|----------------|----------------| ----------------|
| ![Status](https://img.shields.io/badge/Status-Under_Development-red) | ![License](https://img.shields.io/badge/License-MIT-blue) | ![Language](https://img.shields.io/badge/Language-Python-yellow) | ![Release](https://img.shields.io/badge/Release-v1.0.0-green) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4740735.svg)](https://github.com/LangeLab/Analysis_of_QuEStVar_Manuscript) | [![Citation](https://img.shields.io/badge/Citation-Not_Available-lightgrey)](https://github.com/LangeLab/Analysis_of_QuEStVar_Manuscript) |

This repository contains the code and data supporting the manuscript "Statistical Testing for Protein Equivalence Identifies Core Functional Modules Conserved Across 360 Cancer Cell Lines and Presents a General Approach to Investigating Biological Systems". Here, you'll find the scripts and Jupyter Notebooks used for:

- **Spike-in Data Analysis:** Preprocessing and applying QuEStVar to a benchmark dataset, demonstrating the method's capabilities.
- **Simulation Studies:** Evaluating QuEStVar's sample equivalence index metric's performance compared to correlation under various simulated scenarios.
- **Cancer Cell Line Analysis:** Using QuEStVar to explore quantitative protein stability and variability to identify conserved functional modules across a large collection of cancer cell lines.

## Repository Structure

- **2022_Frohlich:**
    - Notebooks for spike-in dataset analysis (data prep, QuEStVar application, simulation comparison).
    - Subfolders for data (`raw`, `processed`, `results`, `supplementary`) and figures.
- **2022_Goncalves:**
    - Notebooks for cancer cell line analysis (data prep, statistical testing, stability analysis).
    - Subfolders for data (`raw`, `processed`, `results`, `supplementary`) and figures.
- **Misc:**
    - Notebook describing libraries, functions, and software versions used.
- **questvar:**
    - Source code for the QuEStVar statistical testing framework.
- **supp_notebooks:**
    - HTML versions of notebooks (generated using `nb_to_html.sh` script).
- **requirement.txt, LICENSE, README.md, .gitignore, nb_to_html.sh**

> **Note:** The `data` and `figures` folders are ignored by git to avoid storing large files. The raw data to be placed in the `data/raw` folders can be obtained from the zenodo link provided above.

## How to Use

1. **Clone the repository.**
2. **Install dependencies** 
   - **Create a virtual environment:** It's highly recommended to work in an isolated virtual environment to avoid conflicts. Here's how to create one:
      - **conda:**
         ```bash
         conda create --name my_env python=3.9  # Replace 'my_env' with your desired name
         conda activate my_env
         ```
      - **pip:**
         ```bash
         python3 -m venv my_env  # Replace 'my_env' with your desired name
         source my_env/bin/activate  # Linux/macOS
         my_env\Scripts\activate    # Windows
         ```
   - **Install packages:**
      - **Using `requirements.txt` (if provided):** 
         ```bash
         pip install -r requirements.txt 
         ```
      - **Manually (if `requirements.txt` is not provided):**
        ```bash
        conda install <package_name>  # Use conda for each package
        # OR
        pip install <package_name>    # Use pip for each package
        ```
3. **Explore the Jupyter Notebooks** in the `2022_Frohlich` and `2022_Goncalves` folders to follow the analyses.
4. **Refer to the `questvar` folder** for the core implementation of the QuEStVar method.
