# SentiNet - Phase 1: Data Understanding & Preprocessing

## 📋 Overview

Phase 1 establishes the foundation for SentiNet by comprehensively analyzing and preparing the TII-SSRC-23 dataset for model training. This phase ensures data quality, identifies key features for real-time deployment, and generates actionable insights through advanced visualizations.

---

## 🎯 Objectives Achieved

✅ **Data Loading & Inspection**
- Automated dataset loading with error handling
- Memory usage optimization
- Initial data structure analysis

✅ **Exploratory Data Analysis**
- Attack type distribution analysis
- Feature categorization for real-time capture
- Statistical profiling of all features
- Missing value and data quality assessment

✅ **Data Preprocessing**
- Intelligent missing value handling
- Infinite value replacement
- Feature correlation analysis
- Label encoding with mapping preservation

✅ **Feature Engineering**
- Feature normalization using StandardScaler
- Feature importance calculation (variance + separability)
- Real-time feature mapping for packet capture

✅ **Dataset Balancing**
- Hybrid SMOTE + undersampling strategy
- Class distribution optimization
- Configurable sampling strategies

✅ **Train/Validation/Test Splitting**
- Stratified splitting to maintain class distribution
- 70% training, 10% validation, 20% test (configurable)

✅ **Comprehensive Visualizations**
- 7 detailed visualization reports
- PCA and t-SNE dimensionality reduction
- Feature importance analysis
- Data quality metrics

---

## 📁 File Structure

```
sentinet-desktop/
├── TII-SSRC-23/
│   └── data.csv                          # Raw dataset (place here)
├── src/
│   ├── preprocessing.py                  # Main preprocessing pipeline
│   ├── eda_visualizations.py            # Advanced visualization suite
│   └── utils/
│       └── feature_mapping.py           # Real-time feature extraction
├── processed_data/                       # Generated outputs
│   ├── dataset_splits.npz               # Train/val/test splits
│   ├── scaler.pkl                       # Fitted StandardScaler
│   ├── attack_mapping.json              # Label encoding mapping
│   ├── feature_mapping.json             # Feature metadata
│   ├── top_features.json                # Most important features
│   ├── preprocessing_report.json        # Pipeline summary
│   ├── high_correlations.txt            # Correlated feature pairs
│   └── visualizations/                  # All generated plots
│       ├── class_distribution.png
│       ├── correlation_heatmap.png
│       ├── feature_distributions.png
│       ├── attack_wise_features.png
│       ├── pca_analysis.png
│       ├── tsne_visualization.png
│       ├── feature_importance.png
│       ├── class_balance_detailed.png
│       └── data_quality_report.png
└── README_PHASE1.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

### Step 1: Place Your Dataset

```bash
# Ensure your TII-SSRC-23 dataset is in the correct location
sentinet-desktop/
└── TII-SSRC-23/
    └── data.csv
```

### Step 2: Run Preprocessing Pipeline

```python
# Basic usage
python src/preprocessing.py
```

```python
# Custom configuration
from preprocessing import TIIDataPreprocessor

preprocessor = TIIDataPreprocessor(
    data_path='TII-SSRC-23/data.csv',
    output_dir='processed_data'
)

# Run with custom parameters
preprocessor.run_full_pipeline(
    balance_strategy='hybrid',  # 'smote', 'undersample', or 'hybrid'
    test_size=0.2               # Test set ratio
)
```

### Step 3: Generate Advanced Visualizations

```python
python src/eda_visualizations.py
```

### Step 4: Review Outputs

1. **Check preprocessing report**: `processed_data/preprocessing_report.json`
2. **Review visualizations**: `processed_data/visualizations/`
3. **Examine feature mapping**: `processed_data/feature_mapping.json`
4. **Verify data splits**: `processed_data/dataset_splits.npz`

---

## 📊 Understanding the Outputs

### 1. Dataset Splits (`dataset_splits.npz`)

Contains preprocessed and split data ready for model training:

```python
import numpy as np

data = np.load('processed_data/dataset_splits.npz')
X_train = data['X_train']  # Training features
y_train = data['y_train']  # Training labels
X_val = data['X_val']      # Validation features
y_val = data['y_val']      # Validation labels
X_test = data['X_test']    # Test features
y_test = data['y_test']    # Test labels
```

### 2. Attack Mapping (`attack_mapping.json`)

Maps human-readable attack names to encoded labels:

```json
{
  "Normal": 0,
  "DDoS": 1,
  "DoS": 2,
  "Port Scan": 3,
  "Brute Force": 4,
  ...
}
```

### 3. Feature Mapping (`feature_mapping.json`)

Documents all features and their categories for real-time extraction:

```json
{
  "all_features": ["pkt_len", "protocol", "src_port", ...],
  "feature_categories": {
    "IP Layer": ["src_ip", "dst_ip", "ttl", ...],
    "TCP/UDP": ["src_port", "dst_port", "tcp_flags", ...],
    ...
  },
  "label_column": "label"
}
```

### 4. Feature Importance (`top_features.json`)

Lists most discriminative features:

```json
{
  "top_variance": [...],
  "top_separability": [...],
  "top_combined": [...]
}
```

---

## 🔍 Key Insights from Visualizations

### Class Distribution
- **File**: `class_distribution.png`
- **Insights**: Shows attack type frequencies; identifies minority classes

### Feature Correlations
- **File**: `correlation_heatmap.png`
- **Insights**: Identifies redundant features (correlation > 0.95)
- **Action**: Consider dropping highly correlated features to reduce model complexity

### PCA Analysis
- **File**: `pca_analysis.png`
- **Insights**: 
  - How many components explain 95% variance
  - Feature space separability
  - Potential for dimensionality reduction

### t-SNE Visualization
- **File**: `tsne_visualization.png`
- **Insights**: Non-linear attack type clustering; visual assessment of class separability

### Feature Importance
- **File**: `feature_importance.png`
- **Insights**: Top features ranked by:
  - Variance (information content)
  - Class separability (discrimination power)
  - Combined score (overall importance)

### Data Quality Report
- **File**: `data_quality_report.png`
- **Insights**: 
  - Feature value ranges
  - Outlier distribution
  - Sparsity patterns
  - Imbalance metrics

---

## ⚙️ Configuration Options

### Preprocessing Pipeline

```python
preprocessor = TIIDataPreprocessor(
    data_path='TII-SSRC-23/data.csv',  # Dataset location
    output_dir='processed_data'         # Output directory
)

# Balancing strategies
preprocessor.balance_dataset(
    strategy='hybrid',      # 'smote', 'undersample', 'hybrid'
    sampling_ratio=0.5      # SMOTE sampling ratio (hybrid only)
)

# Train/test split
preprocessor.split_dataset(
    test_size=0.2,          # Test set ratio (20%)
    val_size=0.1            # Validation set ratio (10%)
)
```

### Visualization Suite

```python
eda = AdvancedEDA(
    data_path='processed_data/dataset_splits.npz',
    mapping_path='processed_data/attack_mapping.json',
    feature_path='processed_data/feature_mapping.json',
    output_dir='processed_data/visualizations'
)

# Generate specific visualizations
eda.plot_feature_distributions(n_features=16)
eda.plot_pca_analysis(n_components=2)
eda.plot_tsne_visualization(perplexity=30, n_samples=5000)
eda.plot_feature_importance_proxy(n_top=20)
```

---

## 🔧 Troubleshooting

### Issue: Out of Memory

```python
# Solution 1: Sample data during visualization
eda.plot_tsne_visualization(n_samples=3000)  # Reduce from default 5000

# Solution 2: Reduce feature correlation sample size
# Edit preprocessing.py line with corr calculation
```

### Issue: Missing Dataset

```
Error: FileNotFoundError: TII-SSRC-23/data.csv
```

**Solution**: Ensure dataset is placed in correct directory:
```bash
mkdir -p TII-SSRC-23
# Place data.csv in TII-SSRC-23/
```

### Issue: Label Column Not Detected

**Solution**: Manually specify label column:
```python
preprocessor.label_col = 'your_label_column_name'
```

---

## 📈 Performance Metrics

### Processing Times (approximate)

| Operation | Time (10K samples) | Time (100K samples) | Time (1M samples) |
|-----------|-------------------|---------------------|-------------------|
| Loading | <1s | 2-5s | 10-30s |
| EDA | 1-2s | 5-10s | 30-60s |
| Preprocessing | 2-5s | 10-20s | 60-120s |
| Balancing | 5-10s | 30-60s | 5-10min |
| PCA | 1-2s | 5-10s | 30-60s |
| t-SNE | 10-30s | 2-5min | 15-30min |

*Times vary based on hardware and feature count*

### Memory Requirements

- **Small dataset** (<50K samples): 2-4 GB RAM
- **Medium dataset** (50K-500K): 4-8 GB RAM
- **Large dataset** (>500K): 8-16 GB RAM

---

## 🎓 Feature Categories for Real-Time Capture

The preprocessing pipeline categorizes features based on real-time extractability:

### ✅ Directly Extractable (from packet headers)
- **IP Layer**: src_ip, dst_ip, ip_len, ttl, protocol
- **TCP/UDP**: src_port, dst_port, tcp_flags, tcp_len, udp_len
- **Packet Stats**: pkt_len, header_len, payload_bytes

### ⏱️ Flow-Based (requires tracking)
- **Duration**: flow_duration, active_time, idle_time
- **Counts**: fwd_pkts, bwd_pkts, flow_pkts, syn_count
- **Bytes**: fwd_bytes, bwd_bytes, flow_bytes
- **Timing**: iat_mean, iat_std, iat_max, iat_min

### 🧮 Statistical (computed from flow)
- **Rates**: pkt_rate, byte_rate, flow_rate
- **Ratios**: fwd_pkt_ratio, bwd_pkt_ratio
- **Aggregates**: mean, std, min, max of various metrics

---

## ✅ Phase 1 Completion Checklist

- [ ] Dataset placed in `TII-SSRC-23/data.csv`
- [ ] Preprocessing pipeline executed successfully
- [ ] All output files generated in `processed_data/`
- [ ] Visualizations reviewed and interpreted
- [ ] Feature mapping validated for real-time compatibility
- [ ] Data quality issues identified and addressed
- [ ] Top features noted for model training
- [ ] Class imbalance handling verified

---

## 🔜 Next Steps: Phase 2

Once Phase 1 is complete, proceed to **Phase 2: Model Building & Optimization**:

1. **Model Selection**: Train multiple algorithms (RF, LightGBM, XGBoost, CNN)
2. **Hyperparameter Tuning**: Optimize best-performing models
3. **Ensemble Methods**: Combine models if beneficial
4. **Performance Evaluation**: Accuracy, F1-score, inference speed
5. **Model Export**: Save best model for deployment

---

## 📞 Support & Documentation

### Common Questions

**Q: What if my dataset has different column names?**
A: The preprocessor auto-detects common naming patterns. For custom names, modify `label_candidates` in `preprocessing.py`.

**Q: Can I use a different balancing strategy?**
A: Yes! Options: `'smote'` (oversample minority), `'undersample'` (downsample majority), `'hybrid'` (both).

**Q: How do I add custom features?**
A: Edit `feature_mapping.py` and add your feature extraction logic to the `extract_from_packet()` method.

**Q: Can I run this on a subset of data for testing?**
A: Yes! Sample your CSV before running preprocessing:
```python
df = pd.read_csv('TII-SSRC-23/data.csv')
df_sample = df.sample(n=10000, random_state=42)
df_sample.to_csv('TII-SSRC-23/data.csv', index=False)
```