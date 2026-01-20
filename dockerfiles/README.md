### Example: `train.dockerfile`
Build dockerfile:
```sh
docker build -f dockerfiles/train.dockerfile . -t train:latest
```

Quick run:
```sh
docker run --rm --name train_run \
    -v .models:/models/ \
    -v .reports/figures:/reports/figures/ \
    train:latest
```
