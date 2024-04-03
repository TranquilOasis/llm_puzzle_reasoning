
# LLM Puzzle Reasoning Project

Welcome to the LLM Puzzle Reasoning project. This project aims to leverage large language models (LLMs) for solving various puzzles, including crosswords, Sudoku, word searches, and mazes.

## Structure

The project is organized as follows:

### data/
This directory contains all the datasets. It is divided into:

- **puzzles/**: Different types of puzzles (crossword, sudoku, word search, mazes).
- **processed/**: Preprocessed data ready for training.
- **raw/**: Unmodified datasets.

### models/
Holds the definitions of your models, including:

- **bert_base.py**: For the base BERT model.
- **custom_models.py**: Modifications or new models development.

### notebooks/
Jupyter notebooks for exploratory data analysis, model prototyping, and any other experimental analyses.

### src/
Source code for the project, covering data preprocessing, dataset loading utilities, model training scripts, and other support utilities.

### utilities/
Contains common utilities and helpers, like custom tokenizers and evaluation metrics.

### tests/
Unit and integration tests for your preprocessing scripts, model sanity checks, and other critical functions.

## Setup

To get started with the project, ensure you have the necessary Python environment and dependencies installed. Install all project dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

Provide a brief example of how to run a model training or data preprocessing script here. This will be helpful for both collaboration and future reference.

## Contributing

Contributions to the LLM Puzzle Reasoning project are welcome. Please ensure to follow best practices and guidelines for contributing to the project.

