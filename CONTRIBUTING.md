# Contributing to Daria-Former

## Development Setup

```bash
git clone https://github.com/your-username/Daria-Former.git
cd Daria-Former
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Python 3.10+
- Type hints for all public APIs
- Docstrings for all public classes and methods
- Follow existing code patterns in the project

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit with a clear message
6. Open a Pull Request

## Architecture Guidelines

When adding new components:

- All neural modules should be `nn.Module` subclasses
- Config parameters go in `DariaFormerConfig`
- New layers should support the ESV (Emotion State Vector) interface
- LoRA hooks should be considered for any new linear projection
- Add unit tests for new modules in `tests/`

## Reporting Issues

Open an issue on GitHub with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, PyTorch version, GPU)
