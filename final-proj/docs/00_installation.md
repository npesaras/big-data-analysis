# Installation Guide

This document outlines the installation and setup procedures for the Diabetes Classification System.

## Prerequisites

- Python 3.13 or higher
- Operating System: Linux, macOS, or Windows
- Git version control system

## Package Management

This project utilizes `uv`, a high-performance Python package installer and resolver.

### Installation Steps

**Install uv**

Linux and macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Clone Repository**

```bash
git clone <repository-url>
cd big-data-analysis/final-proj
```

**Install Dependencies**

```bash
uv pip install -r requirements.txt
```

**Launch Application**

```bash
streamlit run main.py
```
