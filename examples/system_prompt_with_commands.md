Search with `grep -rn "<name>" --include="*.py" . | head -40`.
Run the related tests with `python -m pytest tests/test_core.py -q -x 2>&1 | tail -30`.
Check your change with `git diff`.
