# Contributing to bpcontext

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch: `git checkout -b my-feature`
4. Make your changes
5. Run tests: `cargo test`
6. Run clippy: `cargo clippy`
7. Format code: `cargo fmt`
8. Commit and push
9. Open a pull request

## Development Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/bpcontext.git
cd bpcontext

# Build
cargo build

# Run tests
cargo test

# Run with clippy warnings
cargo clippy -- -W warnings
```

## Guidelines

- Follow existing code patterns and style
- Add tests for new functionality
- Keep commits focused and descriptive
- Update documentation for user-facing changes

## Reporting Issues

- Use GitHub Issues
- Include steps to reproduce
- Include your Rust version (`rustc --version`)
- Include your OS and platform

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
