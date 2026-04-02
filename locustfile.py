from locust import HttpUser, between, task


class PredictUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        self.image_bytes = None
        with open("sample.jpg", "rb") as f:
            self.image_bytes = f.read()

    @task
    def predict(self):
        files = {"file": ("sample.jpg", self.image_bytes, "image/jpeg")}
        self.client.post("/predict", files=files, timeout=30)
