"""
SentiNet - Model Testing & Verification Script
Test your trained models before deployment

This script helps you:
1. Verify model loads correctly
2. Test predictions on sample data
3. Check inference speed
4. Validate model performance
"""

import joblib
import numpy as np
import json
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_model_and_data(model_path='models/tii_ssrc23_best_model_v1.pkl'):
    """Load trained model and test data"""
    print("="*80)
    print("LOADING MODEL AND DATA")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        model = joblib.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None, None, None, None
    
    # Load test data
    print("\nLoading test data...")
    try:
        data = np.load('processed_data/dataset_splits.npz')
        X_test = data['X_test']
        y_test = data['y_test']
        print(f"✓ Test data loaded: {X_test.shape[0]:,} samples")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None, None, None, None
    
    # Load scaler
    print("\nLoading scaler...")
    try:
        scaler = joblib.load('processed_data/scaler.pkl')
        print("✓ Scaler loaded successfully")
    except Exception as e:
        print(f"✗ Error loading scaler: {e}")
        scaler = None
    
    # Load attack mapping
    print("\nLoading attack mapping...")
    try:
        with open('processed_data/attack_mapping.json', 'r') as f:
            attack_mapping = json.load(f)
        label_to_attack = {v: k for k, v in attack_mapping.items()}
        print(f"✓ Attack mapping loaded: {len(attack_mapping)} classes")
    except Exception as e:
        print(f"✗ Error loading mapping: {e}")
        label_to_attack = {}
    
    return model, X_test, y_test, scaler, label_to_attack


def test_single_prediction(model, X_test, y_test, label_to_attack):
    """Test prediction on a single sample"""
    print("\n" + "="*80)
    print("SINGLE SAMPLE PREDICTION TEST")
    print("="*80)
    
    # Take first sample
    sample = X_test[0:1]
    true_label = y_test[0]
    
    print(f"\nTrue label: {true_label} ({label_to_attack.get(true_label, 'Unknown')})")
    
    # Predict
    start_time = time.time()
    prediction = model.predict(sample)
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    print(f"Predicted label: {prediction[0]} ({label_to_attack.get(prediction[0], 'Unknown')})")
    print(f"Inference time: {inference_time:.3f} ms")
    
    # Check if correct
    if prediction[0] == true_label:
        print("✓ Prediction is CORRECT")
    else:
        print("✗ Prediction is INCORRECT")
    
    # Get prediction probabilities if available
    try:
        proba = model.predict_proba(sample)
        print(f"\nPrediction confidence: {proba[0][prediction[0]]*100:.2f}%")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(proba[0])[-3:][::-1]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_3_indices, 1):
            print(f"  {i}. {label_to_attack.get(idx, 'Unknown')}: {proba[0][idx]*100:.2f}%")
    except:
        pass


def test_batch_prediction(model, X_test, y_test, n_samples=100):
    """Test predictions on a batch of samples"""
    print("\n" + "="*80)
    print(f"BATCH PREDICTION TEST ({n_samples} samples)")
    print("="*80)
    
    # Take batch
    X_batch = X_test[:n_samples]
    y_batch = y_test[:n_samples]
    
    # Predict
    print(f"\nPredicting on {n_samples} samples...")
    start_time = time.time()
    predictions = model.predict(X_batch)
    total_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_batch, predictions)
    f1 = f1_score(y_batch, predictions, average='weighted')
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Time per sample: {(total_time/n_samples)*1000:.3f} ms")
    print(f"  Throughput: {n_samples/total_time:.0f} samples/second")


def test_full_performance(model, X_test, y_test, label_to_attack):
    """Test model on full test set"""
    print("\n" + "="*80)
    print(f"FULL TEST SET EVALUATION ({len(X_test):,} samples)")
    print("="*80)
    
    # Predict on full test set
    print("\nPredicting on full test set...")
    start_time = time.time()
    predictions = model.predict(X_test)
    total_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Time per sample: {(total_time/len(X_test))*1000:.3f} ms")
    print(f"  Throughput: {len(X_test)/total_time:.0f} samples/second")
    
    # Per-class performance
    print("\n" + "-"*80)
    print("Per-Class Performance:")
    print("-"*80)
    
    attack_names = [label_to_attack.get(i, f"Class_{i}") for i in sorted(label_to_attack.keys())]
    report = classification_report(y_test, predictions, target_names=attack_names)
    print(report)
    
    # Performance summary
    print("\n" + "-"*80)
    print("Performance Summary:")
    print("-"*80)
    
    if accuracy >= 0.99:
        print("✓ EXCELLENT: Accuracy ≥ 99%")
    elif accuracy >= 0.95:
        print("✓ GOOD: Accuracy ≥ 95%")
    elif accuracy >= 0.90:
        print("⚠ ACCEPTABLE: Accuracy ≥ 90%")
    else:
        print("✗ POOR: Accuracy < 90%")
    
    avg_inference_ms = (total_time/len(X_test))*1000
    if avg_inference_ms < 0.1:
        print("✓ EXCELLENT: Inference < 0.1 ms/sample")
    elif avg_inference_ms < 0.5:
        print("✓ GOOD: Inference < 0.5 ms/sample")
    elif avg_inference_ms < 1.0:
        print("✓ ACCEPTABLE: Inference < 1 ms/sample")
    else:
        print("⚠ SLOW: Inference ≥ 1 ms/sample")


def test_model_properties(model):
    """Display model properties and information"""
    print("\n" + "="*80)
    print("MODEL PROPERTIES")
    print("="*80)
    
    # Model type
    model_type = type(model).__name__
    print(f"\nModel Type: {model_type}")
    
    # Try to get feature importances
    if hasattr(model, 'feature_importances_'):
        print(f"Feature Importances: Available ✓")
        n_features = len(model.feature_importances_)
        print(f"Number of Features: {n_features}")
        
        # Top 5 important features
        top_5_indices = np.argsort(model.feature_importances_)[-5:][::-1]
        print("\nTop 5 Most Important Features:")
        
        try:
            with open('processed_data/feature_mapping.json', 'r') as f:
                feature_info = json.load(f)
                feature_names = feature_info['all_features']
            
            for i, idx in enumerate(top_5_indices, 1):
                feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                importance = model.feature_importances_[idx]
                print(f"  {i}. {feature_name}: {importance:.4f}")
        except:
            print("  (Feature names not available)")
    else:
        print(f"Feature Importances: Not available")
    
    # Number of classes
    if hasattr(model, 'classes_'):
        n_classes = len(model.classes_)
        print(f"\nNumber of Classes: {n_classes}")
    
    # Model parameters (for tree-based models)
    if hasattr(model, 'n_estimators'):
        print(f"Number of Estimators: {model.n_estimators}")
    
    if hasattr(model, 'max_depth'):
        print(f"Max Depth: {model.max_depth}")


def interactive_test(model, X_test, y_test, label_to_attack):
    """Interactive testing mode"""
    print("\n" + "="*80)
    print("INTERACTIVE TESTING MODE")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("  1. Test random sample")
        print("  2. Test specific sample by index")
        print("  3. Test batch of samples")
        print("  4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # Random sample
            idx = np.random.randint(0, len(X_test))
            sample = X_test[idx:idx+1]
            true_label = y_test[idx]
            
            print(f"\n--- Random Sample #{idx} ---")
            print(f"True: {label_to_attack.get(true_label, 'Unknown')}")
            
            pred = model.predict(sample)[0]
            print(f"Predicted: {label_to_attack.get(pred, 'Unknown')}")
            
            if pred == true_label:
                print("Result: ✓ CORRECT")
            else:
                print("Result: ✗ INCORRECT")
            
            try:
                proba = model.predict_proba(sample)[0]
                print(f"Confidence: {proba[pred]*100:.2f}%")
            except:
                pass
        
        elif choice == '2':
            # Specific sample
            try:
                idx = int(input(f"Enter sample index (0-{len(X_test)-1}): "))
                if 0 <= idx < len(X_test):
                    sample = X_test[idx:idx+1]
                    true_label = y_test[idx]
                    
                    print(f"\n--- Sample #{idx} ---")
                    print(f"True: {label_to_attack.get(true_label, 'Unknown')}")
                    
                    pred = model.predict(sample)[0]
                    print(f"Predicted: {label_to_attack.get(pred, 'Unknown')}")
                    
                    if pred == true_label:
                        print("Result: ✓ CORRECT")
                    else:
                        print("Result: ✗ INCORRECT")
                else:
                    print("Invalid index!")
            except ValueError:
                print("Invalid input!")
        
        elif choice == '3':
            # Batch test
            try:
                n = int(input("Enter number of samples to test: "))
                if n > 0 and n <= len(X_test):
                    X_batch = X_test[:n]
                    y_batch = y_test[:n]
                    
                    start_time = time.time()
                    preds = model.predict(X_batch)
                    elapsed = time.time() - start_time
                    
                    acc = accuracy_score(y_batch, preds)
                    
                    print(f"\n--- Batch Test ({n} samples) ---")
                    print(f"Accuracy: {acc*100:.2f}%")
                    print(f"Time: {elapsed:.3f} seconds")
                    print(f"Per sample: {(elapsed/n)*1000:.3f} ms")
                else:
                    print("Invalid number!")
            except ValueError:
                print("Invalid input!")
        
        elif choice == '4':
            print("\nExiting interactive mode...")
            break
        
        else:
            print("Invalid choice!")


def main():
    """Main testing function"""
    print("="*80)
    print("SENTINET - MODEL TESTING & VERIFICATION")
    print("="*80)
    
    # Load model and data
    model, X_test, y_test, scaler, label_to_attack = load_model_and_data()
    
    if model is None:
        print("\n❌ Failed to load model or data!")
        print("\nMake sure you have:")
        print("  1. Trained a model (run train_model.py)")
        print("  2. Model saved in models/")
        print("  3. Preprocessed data in processed_data/")
        return
    
    print("\n" + "="*80)
    print("MODEL LOADED SUCCESSFULLY ✓")
    print("="*80)
    
    # Run tests
    print("\n\nRunning comprehensive model tests...\n")
    
    # 1. Model properties
    test_model_properties(model)
    
    # 2. Single prediction test
    test_single_prediction(model, X_test, y_test, label_to_attack)
    
    # 3. Batch prediction test
    test_batch_prediction(model, X_test, y_test, n_samples=100)
    
    # 4. Full performance evaluation
    test_full_performance(model, X_test, y_test, label_to_attack)
    
    # 5. Interactive mode
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE ✓")
    print("="*80)
    
    # Ask if user wants interactive mode
    choice = input("\nEnter interactive testing mode? (y/n): ").strip().lower()
    if choice == 'y':
        interactive_test(model, X_test, y_test, label_to_attack)
    
    print("\n" + "="*80)
    print("MODEL VERIFICATION COMPLETE")
    print("="*80)
    print("\n✅ Your model is ready for deployment!")
    print("\nNext steps:")
    print("  1. Review the performance metrics above")
    print("  2. Ensure accuracy > 95% and inference < 1ms")
    print("  3. Proceed to Phase 3: Real-Time Inference")


if __name__ == "__main__":
    main()