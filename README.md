MXFP4 MoE Kernel – GPU MODE Hackathon Submission!!

This project is a submission for the GPU MODE Hackathon (AMD) focused on optimizing key GPU kernels for LLM inference.

Overview

The goal of the competition is to optimize critical components used in modern large language models (LLMs), including:

MXFP4 GEMM
MLA Decode (Attention)
MXFP4 MoE (Mixture of Experts)

Implemented Kernel: MXFP4 MoE

The implementation follows the standard MoE pipeline:

Input → Gating → Routing → Expert Processing → Output Gathering

Key Features

.Top-1 gating mechanism

.Efficient token routing using NumPy

.Grouped expert execution

.Safe and portable (no CUDA dependencies)

.Compatible with evaluation environments


This repository implements a clean, reliable, and portable version of the MXFP4 MoE kernel, designed to run safely in the evaluation environment.
