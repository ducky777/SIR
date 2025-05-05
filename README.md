# SIR: Self-Iterating Retriever

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

SIR (Self-Iterating Retriever) is a lightweight, cost-effective system for multi-hop question answering. SIR enables your retrieval system to incrementally gather information across multiple retrieval steps to answer complex queries that require connecting information from different sources.

## Requirements
python>=3.10

## Installation
`pip install -r requirements.txt`

## 🤖 Our Specialized Lightweight Model

The core advantage of SIR is our specialized, compact model specifically optimized for multi-hop retrieval:

- **Small & Efficient**: Only 363M parameters (compared to 7B+ for LLM-based retrievers)
- **Extended Context**: 8K token context window fine tuned from `"embaas/sentence-transformers-e5-large-v2"`
- **Fine-Tuned Specifically**: Trained on diverse multi-hop reasoning tasks to connect information efficiently

**⭐️ Recommended**: Use our specialized model for optimal performance:
`https://huggingface.co/duckduckpuck/sir-sbert-e5-large-v1`

## Quick Start
`python demo.py`

## Input Format
```python
context = "".join(f"\n{j + 1}. {s}" for j, s in enumerate(memory_documents))
query_with_context = f"Query: {query}\n\nContext:\n{context}"
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Citation

If you use SIR in your research, please cite:

```bibtex
@software{sir2025,
  author = {Ngui Seng Wee},
  title = {SIR: Self-Iterating Retriever for Multi-hop Question Answering},
  year = {2025},
  url = {https://github.com/ducky777/SIR}
}
```