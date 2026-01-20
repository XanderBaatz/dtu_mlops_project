import io
import random
from locust import HttpUser, task, between
from torchvision.datasets import FashionMNIST

class FashionMNISTAPIUser(HttpUser):
    wait_time = between(0.5, 2)  # Random delay between requests

    def on_start(self):
        # Load Fashion-MNIST test dataset once per user
        self.dataset = FashionMNIST(root="././data/fashion_mnist", train=False, download=True)
        self.correct = 0
        self.total = 0

    @task
    def predict(self):
        # Pick a random sample
        idx = random.randint(0, len(self.dataset) - 1)
        image, label = self.dataset[idx]

        # Convert PIL image to PNG in memory
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        # Send request to the API
        with self.client.post(
            "/predict",
            files={"file": ("fashion.png", buf, "image/png")},
            name="/predict (API)",
            catch_response=True
        ) as response:

            # Fail only on real API issues
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                return

            data = response.json()
            if not all(k in data for k in ("class_id", "class_name", "confidence")):
                response.failure("Invalid API response format")
                return

            # Track accuracy separately (does not fail the request)
            self.total += 1
            if data["class_id"] == label:
                self.correct += 1

    def on_stop(self):
        # Print accuracy after load test
        if self.total > 0:
            acc = self.correct / self.total
            print(f"\nModel accuracy under load: {acc:.3f}")