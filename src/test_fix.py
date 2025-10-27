"""
Test the fixed feature extraction
"""
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('models/tii_ssrc23_best_model_v1.pkl')
scaler = joblib.load('processed_data/scaler.pkl')

print("="*80)
print("TESTING FIXED FEATURE EXTRACTION")
print("="*80)

# Simulate normal HTTPS traffic (benign)
normal_features = [
    1.5,      # duration
    25,       # packet count
    15000,    # total bytes
    600,      # avg packet len
    100,      # std packet len
    1200,     # max packet len
    200,      # min packet len
    16.7,     # packet rate
    10000,    # byte rate
    0.04,     # syn ratio (low - good)
    0.96,     # ack ratio (high - good)
    0.3,      # psh ratio
    0,        # rst ratio
    0,        # fin ratio
    0,        # urg ratio
    0.06,     # avg iat
    0.02,     # std iat
    0.15,     # max iat
    0.01,     # min iat
    6,        # protocol (TCP)
    1,        # web port indicator (443)
    0         # NOT remote access port
]

# Pad with small random noise
while len(normal_features) < 79:
    normal_features.append(np.random.uniform(-0.1, 0.1))

normal_features = np.array(normal_features).reshape(1, -1)

# Simulate attack traffic (SYN flood)
attack_features = [
    0.1,      # very short duration
    500,      # MANY packets
    25000,    # lots of bytes
    50,       # small packets
    10,       # low std
    60,       # small max
    40,       # small min
    5000,     # VERY high packet rate (attack!)
    250000,   # VERY high byte rate (attack!)
    0.95,     # VERY high SYN ratio (attack!)
    0.05,     # low ACK ratio
    0,        # no PSH
    0,        # no RST
    0,        # no FIN
    0,        # no URG
    0.0002,   # VERY low IAT (flooding!)
    0.0001,   # low std
    0.001,    # small max
    0.0001,   # tiny min
    6,        # TCP
    0,        # not web port
    0         # not remote access
]

# Pad with small random noise
while len(attack_features) < 79:
    attack_features.append(np.random.uniform(-0.1, 0.1))

attack_features = np.array(attack_features).reshape(1, -1)

print("\n1️⃣ NORMAL HTTPS TRAFFIC (should be Benign):")
print("-"*80)
print(f"Raw features (first 22): {normal_features[0][:22]}")
scaled = scaler.transform(normal_features)
print(f"Scaled (first 22): {scaled[0][:22]}")
pred = model.predict(scaled)[0]
prob = model.predict_proba(scaled)[0]
print(f"✅ Prediction: {pred} ({'Benign' if pred == 0 else 'Malicious'})")
print(f"   Confidence: {prob[pred]*100:.2f}%")
print(f"   Probabilities: Benign={prob[0]*100:.2f}%, Malicious={prob[1]*100:.2f}%")

print("\n\n2️⃣ SYN FLOOD ATTACK (should be Malicious):")
print("-"*80)
print(f"Raw features (first 22): {attack_features[0][:22]}")
scaled = scaler.transform(attack_features)
print(f"Scaled (first 22): {scaled[0][:22]}")
pred = model.predict(scaled)[0]
prob = model.predict_proba(scaled)[0]
print(f"🚨 Prediction: {pred} ({'Benign' if pred == 0 else 'Malicious'})")
print(f"   Confidence: {prob[pred]*100:.2f}%")
print(f"   Probabilities: Benign={prob[0]*100:.2f}%, Malicious={prob[1]*100:.2f}%")

print("\n" + "="*80)
print("✅ If normal traffic shows 'Benign', the fix worked!")
print("="*80)
