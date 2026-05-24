<!-- markdownlint-disable MD033 MD036 MD041 MD045 -->
<p align="center">
    <img src="assets/logo-readme.svg" alt="QuEStVar manuscript analysis" width="320" />
</p>

<p align="center">
    <strong>Reproducible analysis companion for the published QuEStVar manuscript</strong>
</p>

<p align="center">
    Code, notebooks, and reproducible workflows for the QuEStVar manuscript and published analysis archive.
</p>

<p align="center">
    <a href="https://pubs.acs.org/doi/10.1021/acs.jproteome.4c00131"><img src="https://img.shields.io/badge/paper-J._Proteome_Research_2024-0f766e?style=for-the-badge" alt="Journal of Proteome Research 2024" /></a>
    <a href="https://doi.org/10.1021/acs.jproteome.4c00131"><img src="https://img.shields.io/badge/doi-10.1021%2Facs.jproteome.4c00131-0ea5e9?style=for-the-badge" alt="DOI 10.1021/acs.jproteome.4c00131" /></a>
        <a href="https://doi.org/10.5281/zenodo.10694635"><img src="https://img.shields.io/badge/data-Zenodo-2563eb?style=for-the-badge" alt="Zenodo" /></a>
</p>

<p align="center">
        <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-f59e0b?style=for-the-badge" alt="License MIT" /></a>
    <img src="https://img.shields.io/badge/python-3.9.18-0f172a?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9.18" />
    <img src="https://img.shields.io/badge/notebooks-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter notebooks" />
</p>

<p align="center">
    <a href="https://github.com/eneskemalergin/QuEStVar/releases/tag/v0.1.0"><img src="https://img.shields.io/badge/standalone_package-v0.1.0-16a34a?style=for-the-badge" alt="Standalone package v0.1.0" /></a>
    <a href="https://pypi.org/project/questvar/"><img src="https://img.shields.io/badge/PyPI-questvar-1d4ed8?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI questvar" /></a>
</p>

<p align="center">
    <a href="#overview">Overview</a> • <a href="#manuscript">Manuscript</a> • <a href="#standalone-package">Standalone package</a> • <a href="#repository-structure">Repository structure</a> • <a href="#getting-started">Getting started</a> • <a href="#citation">Citation</a>
</p>

## Overview

This repository contains the code and data supporting the published manuscript *Statistical Testing for Protein Equivalence Identifies Core Functional Modules Conserved across 360 Cancer Cell Lines and Presents a General Approach to Investigating Biological Systems*. It preserves the analysis notebooks, manuscript-era source code, and reproducibility scaffolding used for the study.

Use this repository when you want to inspect, rerun, or audit the scientific analysis behind the paper. Use the standalone QuEStVar package when you want the maintained installable software for new analyses.

The repository covers three main analysis streams:

- **Spike-in data analysis.** Preprocessing and applying QuEStVar to a benchmark dataset to demonstrate the method in a controlled setting.
- **Simulation studies.** Evaluating QuEStVar's sample equivalence index metric against correlation under multiple simulated scenarios.
- **Cancer cell line analysis.** Using QuEStVar to explore quantitative protein stability and variability and identify conserved functional modules across 360 cancer cell lines.

## Manuscript

- **Title:** Statistical Testing for Protein Equivalence Identifies Core Functional Modules Conserved across 360 Cancer Cell Lines and Presents a General Approach to Investigating Biological Systems
- **Authors:** Enes K. Ergin, Junia J. K. Myung, and Philipp F. Lange
- **Journal:** *Journal of Proteome Research* 2024, **23** (6), 2169-2185
- **Resources:**
    - Manuscript: <https://pubs.acs.org/doi/10.1021/acs.jproteome.4c00131>
    - DOI: <https://doi.org/10.1021/acs.jproteome.4c00131>
    - Data archive: <https://doi.org/10.5281/zenodo.10694635>

## Standalone Package

This repository is the manuscript companion and analysis archive. The standalone packaged version of QuEStVar is available separately as the first stable release, **v0.1.0**, with its own repository, documentation, and PyPI distribution.

- GitHub repository: <https://github.com/eneskemalergin/QuEStVar>
- GitHub release: <https://github.com/eneskemalergin/QuEStVar/releases/tag/v0.1.0>
- PyPI package: <https://pypi.org/project/questvar/>
- Install command: `pip install questvar`

The split matters because the environments are different. This manuscript repository is pinned to a reproducibility-oriented Python 3.9 stack in `requirements.txt`, while the standalone package targets modern packaged use and is maintained independently on PyPI and GitHub.

## Repository Structure

- `2022_Frohlich/`
        - Contains the spike-in benchmark analysis notebooks.
        - Includes `Notebook_S1.ipynb`, `Notebook_S2.ipynb`, and `Notebook_S3.ipynb`.
        - The working tree also expects analysis subfolders for data (`raw`, `processed`, `results`, `supplementary`) and figures.
- `2022_Goncalves/`
        - Contains the cancer cell line analysis notebooks.
        - Includes `Notebook_S4.ipynb` through `Notebook_S8.ipynb` together with the corresponding analysis outputs used in the manuscript workflow.
- `Misc/`
        - Contains supporting notebooks such as `Notebook_S9.ipynb` and `Notebook_S10.ipynb`.
        - These document libraries, helper functions, and software-version context used during the study.
- `questvar/`
        - Contains the manuscript-era QuEStVar source code used directly by the notebooks in this repository.
- `supp_notebooks/`
        - Contains HTML-rendered notebook outputs generated from the Jupyter notebooks.
- `nb_to_html.sh`
        - Helper script for converting notebooks into HTML outputs.
- `requirements.txt`
        - Pinned Python dependencies for reproducing the manuscript analyses.
- `.gitignore`, `CITATION.cff`, `LICENSE`, `README.md`
        - Repository metadata, citation metadata, ignore rules, licensing, and top-level documentation.

> [!NOTE]
> The `data` and `figures` folders are intentionally ignored by git to avoid committing large artifacts. Raw input files that belong in the `data/raw` folders should be obtained from the Zenodo archive linked above.

## Getting Started

1. Clone this repository.
2. Create a dedicated Python environment.

   **conda**

   ```bash
   conda create --name my_env python=3.9
   conda activate my_env
   ```

   **venv**

   ```bash
   python3 -m venv my_env
   source my_env/bin/activate
   # On Windows use: my_env\Scripts\activate
   ```

3. Install the pinned manuscript dependencies.

   ```bash
   pip install -r requirements.txt
   ```

    If you prefer manual installation, install the packages listed in `requirements.txt` individually with either `conda install <package_name>` or `pip install <package_name>`.

4. Explore the Jupyter notebooks in `2022_Frohlich/`, `2022_Goncalves/`, and `Misc/` to follow the manuscript analyses end to end.
5. Refer to `questvar/` for the core statistical testing implementation used by the notebooks.
6. Optionally generate HTML notebook exports with `nb_to_html.sh` when you want portable rendered outputs.

## Reproducibility Notes

- The manuscript environment is pinned to Python 3.9.18 and the dependency versions listed in `requirements.txt`.
- The repository is designed around notebook-driven analysis. Large intermediate files and figures are expected to live outside version control.
- The `questvar/` code in this repository reflects the manuscript analysis snapshot rather than the separately maintained installable package.
- Notebook outputs and large derived artifacts are expected to be regenerated locally from the archived workflow and downloaded raw inputs.

## Citation

If you use the manuscript analyses, data workflow, or the scientific framing of QuEStVar, cite the paper:

The repository also includes a root `CITATION.cff` file so GitHub and citation tools can surface the preferred manuscript citation directly.

```bibtex
@article{ergin2024questvar,
  author = {Ergin, Enes K. and Myung, Junia J. K. and Lange, Philipp F.},
  title = {Statistical Testing for Protein Equivalence Identifies Core Functional Modules Conserved across 360 Cancer Cell Lines and Presents a General Approach to Investigating Biological Systems},
  journal = {Journal of Proteome Research},
  year = {2024},
  volume = {23},
  number = {6},
  pages = {2169--2185},
  doi = {10.1021/acs.jproteome.4c00131}
}
```
