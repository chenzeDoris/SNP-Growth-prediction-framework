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

### 2. Genotype Data
The genotype data contains the actual SNP information and classification.

**Required Columns:**
- `Class`: The dependent variable (Y) for analysis - Binary labels only
- `SNP_*`: Individual SNP columns representing independent variables (X)

## Expected Data Format

### Column Structure