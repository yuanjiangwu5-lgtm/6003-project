# Cryptanalytic Extraction of ReLU Neural Networks  
*A Multi-Module Reconstruction of Soft-Label, Hard-Label, and Verification-Guided Attacks*

This repository contains a complete, multi-stage reconstruction of modern
cryptanalytic model-extraction attacks on ReLU neural networks.  
Following the theoretical developments from **CRYPTO 2020**, **EUROCRYPT 2024**, **EUROCRYPT 2025**, and the **2025 ePrint extension**, we implement four progressively more challenging extraction modules—ranging from soft-label probing to hard-label boundary sampling and verification-guided refinement.

The work demonstrates that **ReLU networks inherently leak structural
information** through kink geometry and decision-boundary topology, enabling
polynomial-query, polynomial-time extraction even under restricted query models.

---

## 📦 Repository Structure

6003-project/
│
├── main/ # Module 1: Soft-label kink-based extraction
│
├── Module2&ExtensionA/ # Module 2 + Extension A:
│ │ Structured sampling + adaptive verification loop
│ └── ...
│
├── Module3/ # Module 3: Hard-label boundary-based extraction
│ └── ...
│
└── README.md

---

## 🧩 Overview of Modules

### **Module 1 — Soft-Label Differential Probing (10–10–1 Network)**
Implements the minimal viable attack from **CRYPTO 2020**:
- sample one-dimensional input lines
- detect kinks via slope changes
- convert slope jumps into linear constraints
- recover the full parameters of a (10,10,1) ReLU network

This module establishes the core idea:  
> *Piecewise linearity + kink leakage ⇒ solvable algebraic constraints.*

---

### **Module 2 — Structured Sampling & Polynomial-Time Recovery (EUROCRYPT 2024)**  
A stability-improved extraction framework inspired by the 2024 result:
- structured direction generation  
- direction clustering  
- block-wise solving  
- Tikhonov-regularized least squares  
- significantly lower query complexity  
- improved numerical stability

---

### **Module 3 — Hard-Label Extraction (EUROCRYPT 2025)**  
Implements boundary-only extraction using:
- HopSkipJump decision-boundary sampling
- Foolbox / ART adversarial tools  
- MILP/MIP-based region solving (Marabou-compatible)

Validates the 2025 theoretical insight:  
> *Labels alone are sufficient for functional reconstruction.*

---

### **Extension A — Verification-Guided Adaptive Sampling (ePrint 2025)**  
Adds a refinement loop:
- verification-guided resampling  
- region-wise confidence assessment  
- robustness-improving perturbation patterns  
- reduced query usage with higher precision

This module improves both efficiency and functional equivalence guarantees.

---

## 🎯 Key Insight

Across all modules, we observe a consistent phenomenon:

> **Even as observability decreases—from logits → labels-only → verified labels—the internal structure of ReLU networks continues to leak through kink geometry and boundary topology.**

This leakage enables:
- polynomial-query extraction,
- polynomial-time solving,
- and strong guarantees of functional equivalence (via formal verification).

---
