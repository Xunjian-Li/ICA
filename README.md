This is an impressive project. Given that your MM Estimation Algorithm focuses on speed, GPU acceleration, and mathematical robustness (Fan-von Neumann inequality), your README should highlight those performance gains.

Since you are including the 549.57 MB X_whitened.mat and other data, I have included a section on how users should handle the Large File Storage (LFS) requirements.
An MM Estimation Algorithm for Independent Component Analysis

This repository contains the official Julia and Python implementations of the Majorization-Minimization (MM) algorithm for Independent Component Analysis (ICA), as described in the paper by Xun-Jian Li, Hua Zhou, and Kenneth Lange (UCLA).
🚀 Overview

Estimating the unmixing matrix in ICA usually involves complex optimization. This project introduces a quadratic lower-bound surrogate for the ICA loglikelihood. By applying the Fan-von Neumann inequality, we derive an update rule that:

    Is Computationally Efficient: Dominated by matrix-matrix multiplications and small-scale SVD.

    Supports GPU Acceleration: Highly parallelizable, outperforming Newton-type methods (including Picard) by up to an order of magnitude.

    Guarantees Convergence: Built on the mathematically grounded MM principle.

📂 Repository Structure

    src/: Core Julia implementation (ICAmm.jl, InfomaxICA.jl).

    faster-ica/: Python implementation and benchmark scripts.

    faster-ica/examples/: Real-world datasets (EEG and fMRI data).

    performing ICA.ipynb: Demonstration notebook showing how to apply the algorithm.

🛠 Installation & Requirements
Julia Setup

    Clone the repository.

    Open Julia in the project folder:
    Julia

    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()

Python Setup
Bash

cd faster-ica
pip install -e .

📦 Large File Storage (Git LFS)

This repository uses Git LFS to store large datasets (e.g., X_whitened.mat, eeg.mat). To ensure you download the actual data rather than just the pointer files, run:
Bash

git lfs install
git lfs pull

📊 Benchmarks

Our MM method consistently outperforms Picard's method and other Newton-type algorithms across various datasets.
Method	Convergence Speed	GPU Support	Cost per Iteration
MM (Proposed)	High	Native	2× Matrix Mult + 1× SVD
Picard	Medium-High	Limited	High (Newton-type)
Natural Gradient	Medium	Variable	Matrix Inversion
📖 Citation

If you use this code or algorithm in your research, please cite:

    Li, X. J., Zhou, H., & Lange, K. (2026). An MM Estimation Algorithm for Independent Component Analysis.

✉️ Contact

For questions regarding the MM algorithm or the Julia implementation, please contact Xun-Jian Li at UCLA.
