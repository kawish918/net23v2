"""
Diagnostic tool to check why all traffic is classified as malicious
Analyzes feature extraction and compares with training data distribution
"""

import joblib
import numpy as np
import json

# Load model and scaler
model = joblib.load('models/tii_ssrc23_best_model_v1.pkl')
scaler = joblib.load('processed_data/scaler.pkl')

# Load test data to see what "normal" features look like
test_data = np.load('processed_data/dataset_splits.npz', allow_pickle=True)
X_test = test_data['X_test']
y_test = test_data['y_test']

print("="*80)
print("DIAGNOSTIC: Feature Analysis")
print("="*80)

# Get some benign samples from test set
benign_indices = np.where(y_test == 0)[0][:5]
malicious_indices = np.where(y_test == 1)[0][:5]

print(f"\n📊 BENIGN Traffic Features (from training data):")
print("-"*80)
for i, idx in enumerate(benign_indices[:3], 1):
    sample = X_test[idx]
    print(f"\nBenign Sample {i}:")
    print(f"  First 10 features: {sample[:10]}")
    print(f"  Features 20-30: {sample[20:30]}")
    print(f"  Last 10 features: {sample[-10:]}")
    print(f"  Non-zero count: {np.count_nonzero(sample)}/79")
    print(f"  Mean: {np.mean(sample):.4f}, Std: {np.std(sample):.4f}")

print(f"\n\n🚨 MALICIOUS Traffic Features (from training data):")
print("-"*80)
for i, idx in enumerate(malicious_indices[:3], 1):
    sample = X_test[idx]
    print(f"\nMalicious Sample {i}:")
    print(f"  First 10 features: {sample[:10]}")
    print(f"  Features 20-30: {sample[20:30]}")
    print(f"  Last 10 features: {sample[-10:]}")
    print(f"  Non-zero count: {np.count_nonzero(sample)}/79")
    print(f"  Mean: {np.mean(sample):.4f}, Std: {np.std(sample):.4f}")

print("\n\n🔍 SIMULATED CAPTURED TRAFFIC (what your sniffer creates):")
print("-"*80)

# Simulate what your feature extraction creates
simulated_features = [
    0.5,      # duration
    10,       # packet count
    5000,     # total bytes
    500,      # avg packet len
    50,       # std packet len
    600,      # max packet len
    400,      # min packet len
    20,       # packet rate
    10000,    # byte rate
    0.2,      # syn ratio
    0.8,      # ack ratio
    0.1,      # psh ratio
    0,        # rst ratio
    0,        # fin ratio
    0,        # urg ratio
    0.05,     # avg iat
    0.01,     # std iat
    0.1,      # max iat
    0.01,     # min iat
    6,        # protocol (TCP)
    443,      # src_port (UNSCALED!)
    52341     # dst_port (UNSCALED!)
]

# Pad with zeros like the GUI does
while len(simulated_features) < 79:
    simulated_features.append(0)

simulated_features = np.array(simulated_features).reshape(1, -1)

print(f"\nSimulated 'Normal HTTPS' Traffic:")
print(f"  First 10 features: {simulated_features[0][:10]}")
print(f"  Features 20-30: {simulated_features[0][20:30]}")
print(f"  Last 10 features: {simulated_features[0][-10:]}")
print(f"  Non-zero count: {np.count_nonzero(simulated_features)}/79")
print(f"  Mean: {np.mean(simulated_features):.4f}, Std: {np.std(simulated_features):.4f}")

# Test prediction BEFORE scaling
print(f"\n❌ Prediction on RAW features (WRONG - should fail):")
try:
    pred_raw = model.predict(simulated_features)[0]
    prob_raw = model.predict_proba(simulated_features)[0]
    print(f"  Prediction: {pred_raw} ({'Benign' if pred_raw == 0 else 'Malicious'})")
    print(f"  Confidence: {prob_raw[pred_raw]*100:.2f}%")
    print(f"  Probabilities: Benign={prob_raw[0]*100:.2f}%, Malicious={prob_raw[1]*100:.2f}%")
except Exception as e:
    print(f"  Error: {e}")

# Test prediction AFTER scaling (what GUI does)
scaled_features = scaler.transform(simulated_features)
print(f"\n✅ Prediction on SCALED features (what GUI does):")
print(f"  Scaled first 10 features: {scaled_features[0][:10]}")
print(f"  Scaled features 20-30: {scaled_features[0][20:30]}")
print(f"  Scaled last 10 features: {scaled_features[0][-10:]}")
pred_scaled = model.predict(scaled_features)[0]
prob_scaled = model.predict_proba(scaled_features)[0]
print(f"  Prediction: {pred_scaled} ({'Benign' if pred_scaled == 0 else 'Malicious'})")
print(f"  Confidence: {prob_scaled[pred_scaled]*100:.2f}%")
print(f"  Probabilities: Benign={prob_scaled[0]*100:.2f}%, Malicious={prob_scaled[1]*100:.2f}%")

print("\n\n🎯 KEY ISSUES:")
print("-"*80)
print("1. ❌ Ports (443, 52341) are UNSCALED - model expects scaled values ~0-3")
print("2. ❌ Last 57 features are ALL ZEROS - model trained on non-zero values")
print("3. ❌ Missing forward/backward flow separation (critical features)")
print("4. ❌ Zero-padding creates abnormal pattern -> classified as malicious")

print("\n\n💡 SOLUTION:")
print("-"*80)
print("Option 1: Don't use src_port/dst_port in features (replace with meaningful values)")
print("Option 2: Scale port numbers properly (but they're categorical, not continuous)")
print("Option 3: Use actual test data samples for validation (not live capture)")
print("Option 4: Retrain model on simplified features without forward/backward separation")

print("\n" + "="*80)
