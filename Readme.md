# Genotype Data Analysis - Data Input Instructions

## Overview

This document describes the required data format and structure for genotype data analysis. The analysis expects CSV input files containing both genotype data and position information for Single Nucleotide Polymorphisms (SNPs).

## Input Data Requirements

### File Format
- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8 recommended
- **File Extension**: `.csv`

## Data Structure

Your input CSV file should contain two main types of data:

### 1. Position Data
The position data provides genomic location information for each SNP.

**Required Columns:**
- `Chromosome` or `Scaffold`: The chromosome or scaffold identifier
- `Position`: Genomic position (integer)
- `Name`: SNP identifier/name
**Example Data**
```csv
Chromosome,Position,Name
chr1,1234567,rs001
chr1,2345678,rs002
chr2,3456789,rs003
```
### 2. Genotype Data
The genotype data contains the actual SNP information and classification.

**Required Columns:**
- `Class`: The dependent variable (Y) for analysis - Binary labels only
- `SNP_*`: Individual SNP columns representing independent variables (X)

**Example Data**
```csv
Class,SNP_001,SNP_002,SNP_003
FAST,0,1,2
FAST,1,0,2
SLOW,2,1,0
```
### 

# Machine Learning Experiment Runner - Usage Instructions

## Overview

This Python script (`runner.py`) is a comprehensive machine learning pipeline designed for genotype/SNP data analysis. It supports multiple classification algorithms, feature selection methods, and experimental configurations for genomic association studies.

## Key Features

- **Multiple ML Algorithms**: SVM, Random Forest, Logistic Regression, Naive Bayes, DWD
- **Feature Selection**: Chi2, CMIM, MI, Relief methods
- **DMFS Integration**: DK Pre-filtering
- **Cross-validation**: Built-in train/test splitting with reproducible results


### 1. Command Line Interface

```bash
# Basic syntax
python runner.py <RandomState> <DMFS> <doGA> [--methods METHOD1 METHOD2 ...]

# Examples
python runner.py 123 False False --methods Chi2
python runner.py 42 True False --methods raw Chi2 CMIM MI Relief
```


## Parameters Explanation

### Command Line Arguments

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `RandomState` | int | Random seed for reproducibility | Any integer (e.g., 42, 123) |
| `DMFS` | bool | Enable DK pre-filtering | True/False |
| `doGA` | bool | Enable Genetic Algorithm optimization | True/False |
| `--methods` | list | Feature selection methods to run | raw, Chi2, CMIM, MI, Relief |

### Feature Selection Methods

- **raw**: No feature selection (use all SNPs)
- **Chi2**: Chi-square test for feature selection
- **CMIM**: Conditional Mutual Information Maximization
- **MI**: Mutual Information
- **Relief**: Relief-based feature selection

### Default Feature Sizes
```python
fs_params = {
    'raw': 0,        # All features
    'Chi2': 4000,    
    'CMIM': 1000,    
    'MI': 1200,      
    'Relief': 1000,  
}
```
