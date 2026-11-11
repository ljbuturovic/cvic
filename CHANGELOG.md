# Changelog

All notable changes to `cvic` are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.3.0] - 2026-11-10

### Added
- `--model` now defaults to `resnet18`, so `--data` is the only required
  argument (e.g. `cvic --data path/to/dataset`).

### Fixed
- Relative `--data` / `--test-data` paths now work. They are resolved to
  absolute paths before Ray Tune launches trials, which run in their own
  working directory. Previously a relative path passed the initial dataset
  scan in the main process but failed inside each trial with
  `FileNotFoundError: '<path>/train'`.
