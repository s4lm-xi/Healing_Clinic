import argparse
import joblib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
args = parser.parse_args()


model = joblib.load('python/model.joblib')
m = np.array([args.input])
m = m.reshape(-1, 1)
pred = model.predict(m)
print(pred[0])