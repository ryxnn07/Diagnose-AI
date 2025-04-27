import unittest
from app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome to the Disease Prediction API!", response.data)

    def test_symptoms_route(self):
        response = self.app.get('/symptoms')
        self.assertEqual(response.status_code, 200)

def test_predict_route(self):
    response = self.app.post('/predict', json={
        "symptoms": {
            "age": 30,
            "gender": "male",
            "fever": "yes",
            "cough": "no",
            "fatigue": "yes",
            "difficulty breathing": "no",
            "blood pressure": "normal",
            "cholesterol level": "high"
        }
    })
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"predicted_diseases", response.data)


if __name__ == '__main__':
    unittest.main()