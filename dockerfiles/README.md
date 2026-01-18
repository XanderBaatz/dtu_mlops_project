### Example: `train.dockerfile`
Build dockerfile:
```sh
docker build -f dockerfiles/train.dockerfile . -t train:latest
```

Quick run:
```
docker run --name train_run --rm \
    -v .models:/models/ \
    -v .reports/figures:/reports/figures/ \
    train:latest
```
