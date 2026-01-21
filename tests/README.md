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

# To test the api with locust run (where the first line is starting the api)
```sh
uv run uvicorn src.dtu_mlops_project.apifile:app --reload
uv run locust -f tests/locustfile.py
```
# then open http://localhost:8089/ in a browser and use http://127.0.0.1:8000 as host (or whatever port you used for running locust and the api).
