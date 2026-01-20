### Example: `train.dockerfile`
Build dockerfile:
```sh
docker build -f dockerfiles/train.dockerfile . -t train:latest
```

Force build for `amd64` architecture (useful for MacOS):
```sh
docker buildx build --platform linux/amd64 -f dockerfiles/train.dockerfile . -t train:latest
```

Quick run:
```
docker run --rm --name train_run \
    -v .models:/models/ \
    -v .reports/figures:/reports/figures/ \
    train:latest
```
