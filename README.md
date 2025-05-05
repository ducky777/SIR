# SIR: Self-Iterating Retriever

[![PyPI version](https://badge.fury.io/py/sir.svg)](https://badge.fury.io/py/sir)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

SIR (Self-Iterating Retriever) is a lightweight, cost-effective system for multi-hop question answering. SIR enables your retrieval system to incrementally gather information across multiple retrieval steps to answer complex queries that require connecting information from different sources.

## Requirements
python>=3.10

## Installation
`pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
`pip install -r requirements.txt`

## 🤖 Our Specialized Lightweight Model

The core advantage of SIR is our specialized, compact model specifically optimized for multi-hop retrieval:

- **Small & Efficient**: Only 363M parameters (compared to 7B+ for LLM-based retrievers)
- **Extended Context**: 8K token context window fine tuned from `"embaas/sentence-transformers-e5-large-v2"`
- **Cost-Effective**: 10-20x cheaper to run than larger models with similar performance
- **Fast Inference**: Low latency even on CPU, no GPU required for reasonable performance
- **Fine-Tuned Specifically**: Trained on diverse multi-hop reasoning tasks to connect information efficiently

**⭐️ Recommended**: Use our specialized model for optimal performance:
`https://huggingface.co/duckduckpuck/sir-sbert-e5-large-v1`

## Quick Start
`python demo.py`

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Citation

If you use SIR in your research, please cite:

```bibtex
@software{sir2023,
  author = {Ngui Seng Wee},
  title = {SIR: Self-Iterating Retriever for Multi-hop Question Answering},
  year = {2025},
  url = {https://github.com/ducky777/sir}
}
```