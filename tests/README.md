# To run the tests run the following in the terminal
```sh
uv run pytest tests/test_model.py -v --tb=short 2>&1 | tail -100
uv run pytest tests/test_data.py -v --tb=short 2>&1 | tail -40
```

# To run coverage run
```sh
uv run coverage run -m pytest tests/
```

or

```sh
uv run coverage report -m
```
