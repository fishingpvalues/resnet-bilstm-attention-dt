# Contributing to ResNet-BiLSTM-Attention Digital Twin

Thank you for your interest in contributing to this project! This document outlines the process for contributing to the ResNet-BiLSTM-Attention Digital Twin framework.

## Code of Conduct

By participating in this project, you agree to abide by common open source practices:
- Be respectful and inclusive in your communications
- Focus on constructive feedback and improvement
- Consider the impact of your contributions on the project's goals

## How to Contribute

### Reporting Issues

If you find bugs or have feature requests:
1. Check the [issue tracker](https://github.com/fishingpvalues/resnet-bilstm-attention-dt/issues) to see if it has already been reported
2. If not, create a new issue with a descriptive title and detailed description
3. Include steps to reproduce, expected vs. actual behavior, and your environment details

### Pull Requests

1. Fork the repository
2. Create a new branch for your changes (`git checkout -b feature/your-feature-name`)
3. Make your changes following the coding standards below
4. Add tests that verify your changes work correctly
5. Update documentation as needed
6. Submit a pull request with a clear description of the changes

### Coding Standards

All contributions must follow our [Coding Style Guide](CODING_STYLE.md). Please read this document carefully before submitting any code changes. The key points:

- Use [pre-commit](https://pre-commit.com/) hooks to automatically format your code
- Follow the Google-style docstring format
- Maintain 100 character line length
- Use type hints where appropriate
- Keep Jupyter notebooks clean (no output cells in commits)

For detailed information about our linting and formatting tools:
1. Review the `.pre-commit-config.yaml` file
2. See the comprehensive `.linter-config.yaml` file
3. Follow the IDE setup instructions in [CODING_STYLE.md](CODING_STYLE.md)

### Development Setup

1. Follow the installation instructions in the README.md
2. Use the UV package manager for dependency management
3. Run tests before submitting changes

## Testing

Run the test suite to ensure your changes don't break existing functionality:

```bash
# For traditional hypothesis tests
python hypothesis_testing.py

# For CCT testing
python run_cct.py --model both --runs 5

# For identical data testing
python run_identical_data_tests.py
```

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT license.