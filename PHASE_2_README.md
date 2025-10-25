# SentiNet - Phase 2: Model Building & Optimization

## 📋 Overview

Phase 2 implements a comprehensive machine learning pipeline with multiple state-of-the-art models for intrusion detection. This phase focuses on achieving the highest possible accuracy, F1-scores, and real-time inference speed.

---

## 🎯 Models Implemented

### Traditional Machine Learning
1. **Random Forest** - Robust ensemble of decision trees
2. **XGBoost** - Gradient boosting with regularization
3. **LightGBM** - Fast, efficient gradient boosting
4. **Gradient Boosting** - Classical boosting algorithm

### Ensemble Methods
5. **Voting Classifier** - Soft/hard voting ensembles
6. **Stacking Classifier** - Meta-learning with multiple base models
7. **Weighted Ensemble** - Performance-weighted predictions

### Deep Learning (Optional)
8. **MLP** - Multi-Layer Perceptron
9. **1D CNN** - Convolutional Neural Network for pattern detection
10. **LSTM** - Long Short-Term Memory for sequential patterns
11. **Attention Model** - Transformer-based with multi-head attention

---

## 📁 File Structure

```
sentinet-desktop/
├── src/
│   ├── train_model.py              # Main model training
│   ├── ensemble_model.py           # Ensemble methods
│   └── deep_learning_model.py      # Neural networks
├── models/                          # Saved models
│   ├── tii_ssrc23_best_model_v1.pkl
│   ├── tii_ssrc23_xgboost_v1.pkl
│   ├── tii_ssrc23_lightgbm_v1.pkl
│   ├── tii_ssrc23_ensemble_*.pkl
│   └── *_metadata.json
├── results/                         # Training results
│   ├── model_comparison.csv
│   ├── ensemble_comparison.csv
│   ├── training_summary.json
│   ├── *_classification_report.txt
│   └── plots/
│       ├── *_confusion_matrix.png
│       ├── *_feature_importance.png
│       ├── model_comparison.png
│       └── ensemble_comparison.png
└── processed_data/                  # From Phase 1
    └── dataset_splits.npz
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Core requirements
pip install numpy pandas scikit-learn xgboost lightgbm joblib matplotlib seaborn

# Optional: Deep Learning
pip install tensorflow  # or tensorflow-gpu for GPU support
```

### Step 1: Train Traditional ML Models

```python
# Run main training script
python src/train_model.py
```

This will:
- Train 4 traditional ML models (RF, XGBoost, LightGBM, GB)
- Perform 5-fold cross-validation
- Generate confusion matrices and feature importance plots
- Save the best model automatically

### Step 2: Train Ensemble Models (Optional but Recommended)

```python
# Run ensemble training
python src/ensemble_model.py
```

This will:
- Create voting ensembles (soft and hard)
- Build stacking classifiers with different meta-learners
- Train weighted ensemble with optimal weights
- Compare all ensemble methods

### Step 3: Train Deep Learning Models (Optional)

```python
# Run deep learning training
python src/deep_learning_model.py
```

This will:
- Train MLP, CNN, LSTM, and Attention models
- Use early stopping and learning rate scheduling
- Generate training history plots
- Save the best performing neural network

---

## 📊 Understanding the Results

### 1. Model Comparison (`results/model_comparison.csv`)

| Model | Test Accuracy | Test F1-Score | Training Time (s) | Inference (ms) |
|-------|--------------|---------------|-------------------|----------------|
| LightGBM | 99.45% | 99.43% | 35.2 | 0.082 |
| XGBoost | 99.38% | 99.36% | 48.7 | 0.095 |
| RandomForest | 99.12% | 99.10% | 67.3 | 1.234 |
| GradientBoosting | 98.87% | 98.85% | 156.8 | 0.342 |

*Example results - actual performance depends on your dataset*

### 2. Best Model Selection Criteria

Models are ranked by:
1. **F1-Score** (primary) - Balanced measure of precision and recall
2. **Accuracy** (secondary) - Overall correctness
3. **Inference Speed** (tertiary) - Real-time deployment feasibility

### 3. Confusion Matrix Interpretation

- **Diagonal values** = Correct predictions
- **Off-diagonal values** = Misclassifications
- Look for patterns: Which attacks are confused with each other?

### 4. Feature Importance

Top features indicate which network characteristics are most discriminative for attack detection. Use these insights for:
- Feature selection in production
- Understanding attack signatures
- Optimizing packet capture

---

## ⚙️ Advanced Usage

### Custom Hyperparameters

```python
from train_model import ModelTrainer

trainer = ModelTrainer()

# Train XGBoost with custom parameters
custom_xgb = trainer.train_xgboost(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.9
)

# Train LightGBM with custom parameters
custom_lgb = trainer.train_lightgbm(
    n_estimators=250,
    max_depth=20,
    learning_rate=0.03,
    num_leaves=50
)
```

### Hyperparameter Tuning

```python
from train_model import HyperparameterTuner

tuner = HyperparameterTuner(trainer)

# Tune XGBoost (20 random search iterations)
best_xgb_params = tuner.tune_xgboost(n_iter=20)

# Tune LightGBM
best_lgb_params = tuner.tune_lightgbm(n_iter=20)

# Retrain with best parameters
trainer.train_xgboost(**best_xgb_params)
trainer.train_lightgbm(**best_lgb_params)
```

### Custom Ensemble Weights

```python
from ensemble_model import EnsembleModelTrainer

ensemble_trainer = EnsembleModelTrainer()

# Create weighted ensemble with custom weights
# [RandomForest weight, XGBoost weight, LightGBM weight]
custom_ensemble = ensemble_trainer.create_weighted_ensemble(
    weights=[0.2, 0.4, 0.4]
)
```

### Deep Learning with GPU

```python
from deep_learning_model import DeepLearningTrainer

# Check GPU availability
import tensorflow as tf
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

# Initialize trainer
dl_trainer = DeepLearningTrainer()

# Train with larger batch size on GPU
dl_trainer.train_all_models(epochs=100, batch_size=512)
```

---

## 🔧 Hyperparameter Tuning Guide

### XGBoost Parameters

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|----------------|
| `n_estimators` | 100-500 | More trees = better fit | 200-300 for balance |
| `max_depth` | 6-15 | Deeper = more complex | 8-12 optimal |
| `learning_rate` | 0.01-0.3 | Slower = better | 0.05-0.1 |
| `subsample` | 0.6-1.0 | Prevents overfitting | 0.8 |
| `colsample_bytree` | 0.6-1.0 | Feature sampling | 0.8 |

### LightGBM Parameters

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|----------------|
| `n_estimators` | 100-500 | Number of trees | 200-300 |
| `max_depth` | 10-30 | Tree depth | 15-20 |
| `num_leaves` | 31-100 | Complexity | 31-50 |
| `learning_rate` | 0.01-0.2 | Learning speed | 0.05 |
| `subsample` | 0.6-1.0 | Row sampling | 0.8 |

### Random Forest Parameters

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|----------------|
| `n_estimators` | 100-500 | Number of trees | 200-300 |
| `max_depth` | 20-50 | Tree depth | 30 |
| `min_samples_split` | 2-10 | Splitting threshold | 5 |
| `max_features` | 'sqrt', 'log2' | Feature sampling | 'sqrt' |

---

## 📈 Performance Optimization Tips

### 1. For Highest Accuracy
- Use **ensemble methods** (stacking or weighted)
- Perform **hyperparameter tuning**
- Train on **full dataset** (no sampling)
- Use **cross-validation** to validate

### 2. For Fastest Inference
- Choose **LightGBM** or **XGBoost**
- Reduce `n_estimators` (e.g., 100-150)
- Limit `max_depth` (e.g., 6-8)
- Use **feature selection** (top 30-50 features)

### 3. For Best Balance
- **LightGBM** with moderate parameters
- `n_estimators=200, max_depth=15`
- Provides ~99% accuracy with <0.1ms inference

### 4. For Production Deployment
- **Weighted ensemble** of top 2-3 models
- Precompute feature statistics
- Use **model quantization** if needed
- Implement **caching** for repeated patterns

---

## 🎓 Model Selection Decision Tree

```
Start
  ├─ Need absolute best accuracy?
  │   ├─ Yes → Use Stacking Ensemble or Weighted Ensemble
  │   └─ No  → Continue
  │
  ├─ Have GPU available?
  │   ├─ Yes → Consider Deep Learning (Attention or CNN)
  │   └─ No  → Continue
  │
  ├─ Need very fast inference (<0.1ms)?
  │   ├─ Yes → Use LightGBM with reduced parameters
  │   └─ No  → Continue
  │
  ├─ Need interpretability?
  │   ├─ Yes → Use Random Forest or XGBoost with feature importance
  │   └─ No  → Use LightGBM
  │
  └─ Default: Use LightGBM (best balance)
```

---

## 📊 Expected Performance Benchmarks

### On TII-SSRC-23 Dataset (typical results)

| Metric | Good | Excellent | Outstanding |
|--------|------|-----------|-------------|
| Accuracy | >95% | >98% | >99% |
| F1-Score | >95% | >98% | >99% |
| Precision | >94% | >97% | >99% |
| Recall | >94% | >97% | >99% |
| Inference | <1ms | <0.5ms | <0.1ms |

### Per-Class Performance Goals

- **Normal traffic**: >99% accuracy (high precision critical)
- **DDoS attacks**: >98% detection (high recall critical)
- **Port Scans**: >97% detection
- **Brute Force**: >96% detection
- **Rare attacks**: >90% detection (harder due to imbalance)

---

## 🔍 Troubleshooting

### Issue: Low F1-Score on Minority Classes

**Symptoms**: Overall accuracy high (>95%) but some classes have F1 < 80%

**Solutions**:
1. Check class balance in training data
2. Increase SMOTE sampling ratio in Phase 1
3. Use `class_weight='balanced'` parameter
4. Try ensemble methods (better for imbalanced data)

### Issue: Overfitting (Train >> Test Performance)

**Symptoms**: Training accuracy >99%, test accuracy <95%

**Solutions**:
1. Reduce model complexity (max_depth, n_estimators)
2. Increase regularization (reg_alpha, reg_lambda)
3. Use cross-validation
4. Add more training data or data augmentation

### Issue: Slow Inference Speed

**Symptoms**: Inference time >1ms per sample

**Solutions**:
1. Use LightGBM instead of Random Forest
2. Reduce number of estimators
3. Perform feature selection (use top 50 features)
4. Enable model quantization

### Issue: Out of Memory During Training

**Solutions**:
1. Reduce batch size (for deep learning)
2. Use data sampling for large datasets
3. Train models sequentially instead of all at once
4. Use `tree_method='hist'` for XGBoost

---

## 💾 Model Deployment Checklist

Before deploying to production:

- [ ] Test accuracy >95% on holdout test set
- [ ] Per-class F1-scores >90% for all critical attack types
- [ ] Inference speed <1ms per sample
- [ ] Model size <500MB for easy loading
- [ ] Cross-validation performed (CV std <0.02)
- [ ] Confusion matrix analyzed for critical misclassifications
- [ ] Feature importance validated
- [ ] Model saved with metadata and version
- [ ] Scaler and preprocessing steps saved
- [ ] Attack mapping verified

---

## 📚 Model Export Formats

### Scikit-learn/XGBoost/LightGBM Models

```python
import joblib

# Load model
model = joblib.load('models/tii_ssrc23_best_model_v1.pkl')

# Load scaler
scaler = joblib.load('processed_data/scaler.pkl')

# Load metadata
import json
with open('models/lightgbm_metadata.json', 'r') as f:
    metadata = json.load(f)
```

### Deep Learning Models

```python
from tensorflow import keras

# Load model
model = keras.models.load_model('models/tii_ssrc23_dl_attention_v1.keras')

# Load metadata
import json
with open('models/dl_attention_metadata.json', 'r') as f:
    metadata = json.load(f)
```

---

## 🎯 Phase 2 Completion Checklist

- [ ] All traditional ML models trained successfully
- [ ] Cross-validation performed (5-fold minimum)
- [ ] Model comparison completed
- [ ] Best model identified and saved
- [ ] Confusion matrices generated and analyzed
- [ ] Feature importance plots created
- [ ] Classification reports saved
- [ ] Ensemble models trained (optional)
- [ ] Deep learning models trained (optional)
- [ ] Performance meets deployment criteria
- [ ] All results saved in results/ directory
- [ ] Model metadata files created

---

## 🔜 Next Steps: Phase 3

Once Phase 2 is complete, proceed to **Phase 3: Real-Time Inference Module**:

1. **Packet Sniffer**: Capture live network traffic
2. **Feature Extraction**: Map packets to model input format
3. **Real-Time Inference**: Run predictions on live data
4. **Performance Optimization**: Ensure real-time speed

---

## 📞 Additional Resources

### Recommended Reading
- XGBoost Documentation: https://xgboost.readthedocs.io/
- LightGBM Documentation: https://lightgbm.readthedocs.io/
- Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html

### Performance Tuning
- For imbalanced datasets: Use SMOTE + class weights
- For feature selection: Use feature importance from tree models
- For ensemble methods: Combine top 3 models for best results

### Citation
If using these models in research, consider citing the TII-SSRC-23 dataset and relevant framework papers.