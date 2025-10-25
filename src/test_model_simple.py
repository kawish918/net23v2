"""
Simple Model Testing - Uses Actual Test Data Samples
Tests the model on real samples from the test set
"""

import numpy as np
import joblib
import json

print("="*80)
print("SIMPLE MODEL TEST - USING REAL TEST DATA")
print("="*80)

# Load model and data
print("\nLoading model and test data...")
model = joblib.load('models/tii_ssrc23_best_model_v1.pkl')
scaler = joblib.load('processed_data/scaler.pkl')

# Load test data (already scaled!)
data = np.load('processed_data/dataset_splits.npz')
X_test = data['X_test']
y_test = data['y_test']

# Load mapping
with open('processed_data/attack_mapping.json', 'r') as f:
    attack_mapping = json.load(f)
label_to_attack = {int(v): k for k, v in attack_mapping.items()}

print(f"✓ Loaded {len(X_test):,} test samples")
print(f"✓ Classes: {label_to_attack}")

# Sample random examples
print("\n" + "="*80)
print("TESTING RANDOM SAMPLES FROM TEST SET")
print("="*80)

np.random.seed(42)

# Test 5 benign and 5 malicious
benign_indices = np.where(y_test == 0)[0]
malicious_indices = np.where(y_test == 1)[0]

test_indices = list(np.random.choice(benign_indices, 5, replace=False)) + \
               list(np.random.choice(malicious_indices, 5, replace=False))

np.random.shuffle(test_indices)

correct = 0
total = len(test_indices)

for i, idx in enumerate(test_indices, 1):
    sample = X_test[idx:idx+1]
    true_label = y_test[idx]
    true_class = label_to_attack[int(true_label)]
    
    # Predict
    pred_label = model.predict(sample)[0]
    pred_class = label_to_attack[int(pred_label)]
    pred_proba = model.predict_proba(sample)[0]
    confidence = pred_proba[pred_label] * 100
    
    is_correct = (pred_label == true_label)
    if is_correct:
        correct += 1
    
    print(f"\n{'='*80}")
    print(f"SAMPLE {i}/{total}")
    print(f"{'='*80}")
    print(f"True Label:      {true_class}")
    print(f"Predicted:       {pred_class}")
    print(f"Confidence:      {confidence:.4f}%")
    print(f"Probabilities:   Benign={pred_proba[0]*100:.4f}%, Malicious={pred_proba[1]*100:.4f}%")
    print(f"Result:          {'✅ CORRECT' if is_correct else '❌ WRONG'}")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Total Samples:   {total}")
print(f"Correct:         {correct}")
print(f"Incorrect:       {total - correct}")
print(f"Accuracy:        {(correct/total)*100:.2f}%")
print("="*80)

if correct == total:
    print("\n🏆 PERFECT SCORE - All samples classified correctly!")
elif correct >= total * 0.9:
    print("\n✅ EXCELLENT - Model performs very well on test data")
else:
    print(f"\n⚠️  {correct}/{total} correct - Some misclassifications detected")
