# To run the tests run the following in the terminal
```sh
uv run pytest tests/test_model.py -v --tb=short 2>&1 | tail -100
uv run pytest tests/test_data.py -v --tb=short 2>&1 | tail -40
```