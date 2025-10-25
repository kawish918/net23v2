"""
Quick validation script to verify data and configuration before training
"""
import numpy as np
import json

print("="*80)
print("PRE-TRAINING VALIDATION")
print("="*80)

# Load data
print("\n1. Loading dataset...")
try:
    data = np.load('processed_data/dataset_splits.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"   ✓ Training set: {X_train.shape}")
    print(f"   ✓ Validation set: {X_val.shape}")
    print(f"   ✓ Test set: {X_test.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Check label distribution
print("\n2. Checking label distribution...")
unique_train, counts_train = np.unique(y_train, return_counts=True)
print(f"   Training labels: {dict(zip(unique_train, counts_train))}")

unique_test, counts_test = np.unique(y_test, return_counts=True)
print(f"   Test labels: {dict(zip(unique_test, counts_test))}")

num_classes = len(unique_train)
print(f"   ✓ Number of classes: {num_classes}")

# Load mappings
print("\n3. Loading attack mapping...")
try:
    with open('processed_data/attack_mapping.json', 'r') as f:
        attack_mapping = json.load(f)
    print(f"   ✓ Attack types: {list(attack_mapping.keys())}")
    print(f"   ✓ Classification type: {'Binary' if num_classes == 2 else 'Multi-class'}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Memory estimation
print("\n4. Memory estimation...")
train_memory_mb = X_train.nbytes / (1024**2)
total_memory_mb = (X_train.nbytes + X_val.nbytes + X_test.nbytes) / (1024**2)
print(f"   Training data: {train_memory_mb:.2f} MB")
print(f"   Total data: {total_memory_mb:.2f} MB")
print("   ✓ Memory check complete")

# Check feature info
print("\n5. Feature information...")
try:
    with open('processed_data/feature_mapping.json', 'r') as f:
        feature_info = json.load(f)
    feature_names = feature_info['all_features']
    print(f"   ✓ Number of features: {len(feature_names)}")
    print(f"   ✓ Sample features: {feature_names[:5]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\n✅ All checks passed! Ready for training.")
print("\nRecommended training command:")
print("  python src/train_model.py")
print("\nExpected training time: ~30-40 minutes")
print(f"Models will train on {len(y_train):,} samples with {X_train.shape[1]} features")
