"""
SentiNet - Phase 1: Data Understanding & Preprocessing
TII-SSRC-23 Dataset Analysis and Preparation Module

Author: SentiNet Development Team
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings
import json
import os

warnings.filterwarnings('ignore')

class TIIDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for TII-SSRC-23 dataset
    Handles loading, cleaning, feature engineering, and balancing
    """
    
    def __init__(self, data_path='TII-SSRC-23/data.csv', output_dir='processed_data'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.attack_mapping = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
    def load_data(self):
        """Load and perform initial inspection of the dataset"""
        print("="*80)
        print("PHASE 1: DATA LOADING & INITIAL INSPECTION")
        print("="*80)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"\n✓ Dataset loaded successfully!")
            print(f"  - Shape: {self.df.shape}")
            print(f"  - Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return True
        except Exception as e:
            print(f"\n✗ Error loading dataset: {e}")
            return False
    
    def explore_data(self):
        """Comprehensive exploratory data analysis"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Basic info
        print("\n1. DATASET OVERVIEW")
        print("-" * 50)
        print(f"Number of samples: {len(self.df):,}")
        print(f"Number of features: {len(self.df.columns)}")
        print(f"\nColumn names:\n{list(self.df.columns)}")
        
        # Data types
        print("\n2. DATA TYPES")
        print("-" * 50)
        print(self.df.dtypes.value_counts())
        
        # Missing values analysis
        print("\n3. MISSING VALUES ANALYSIS")
        print("-" * 50)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        
        if missing_df['Missing_Count'].sum() > 0:
            print(missing_df[missing_df['Missing_Count'] > 0])
        else:
            print("✓ No missing values detected!")
        
        # Identify label column (usually 'label', 'attack_type', 'class', etc.)
        label_candidates = ['label', 'attack_type', 'attack', 'class', 'category', 'Label', 'Attack_type']
        label_col = None
        
        for col in label_candidates:
            if col in self.df.columns:
                label_col = col
                break
        
        if label_col is None:
            # Try to find column with categorical string data
            for col in self.df.columns:
                if self.df[col].dtype == 'object' and self.df[col].nunique() < 50:
                    label_col = col
                    break
        
        if label_col:
            print(f"\n4. ATTACK CATEGORIES (Label column: '{label_col}')")
            print("-" * 50)
            attack_dist = self.df[label_col].value_counts()
            print(f"\nNumber of unique attack types: {len(attack_dist)}")
            print(f"\nAttack distribution:")
            for idx, (attack, count) in enumerate(attack_dist.items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"  {idx}. {attack}: {count:,} ({pct:.2f}%)")
            
            # Store label column name
            self.label_col = label_col
            
            # Visualize class distribution
            self.visualize_class_distribution(attack_dist)
            
        # Statistical summary
        print("\n5. STATISTICAL SUMMARY")
        print("-" * 50)
        print(self.df.describe().T)
        
        # Identify feature types
        self.identify_feature_types()
        
        return True
    
    def identify_feature_types(self):
        """Identify and categorize features for real-time packet capture"""
        print("\n6. FEATURE CATEGORIZATION FOR REAL-TIME CAPTURE")
        print("-" * 50)
        
        # Common network features that can be extracted from live packets
        realtime_features = {
            'IP Layer': ['src_ip', 'dst_ip', 'ip_len', 'ttl', 'protocol', 'ip_flags'],
            'TCP/UDP': ['src_port', 'dst_port', 'tcp_flags', 'tcp_len', 'udp_len'],
            'Packet Stats': ['pkt_len', 'pkt_size', 'payload_bytes', 'header_len'],
            'Flow Features': ['duration', 'fwd_pkts', 'bwd_pkts', 'flow_bytes', 
                            'flow_pkts', 'fwd_bytes', 'bwd_bytes'],
            'Timing': ['iat_mean', 'iat_std', 'iat_max', 'iat_min', 'active_mean', 'idle_mean'],
            'Flags & Counts': ['syn_flag', 'ack_flag', 'psh_flag', 'rst_flag', 'fin_flag']
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if hasattr(self, 'label_col') and self.label_col in numeric_cols:
            numeric_cols.remove(self.label_col)
        
        # Categorize available features
        self.feature_categories = {}
        uncategorized = []
        
        for col in numeric_cols:
            found = False
            col_lower = col.lower()
            
            for category, keywords in realtime_features.items():
                if any(keyword in col_lower for keyword in keywords):
                    if category not in self.feature_categories:
                        self.feature_categories[category] = []
                    self.feature_categories[category].append(col)
                    found = True
                    break
            
            if not found:
                uncategorized.append(col)
        
        # Print categorization
        for category, features in self.feature_categories.items():
            print(f"\n{category}: {len(features)} features")
            print(f"  {', '.join(features[:5])}" + ("..." if len(features) > 5 else ""))
        
        if uncategorized:
            print(f"\nUncategorized: {len(uncategorized)} features")
            print(f"  {', '.join(uncategorized[:5])}" + ("..." if len(uncategorized) > 5 else ""))
        
        self.feature_names = numeric_cols
        
        # Save feature mapping for inference
        feature_info = {
            'all_features': self.feature_names,
            'feature_categories': self.feature_categories,
            'label_column': self.label_col if hasattr(self, 'label_col') else None
        }
        
        with open(f"{self.output_dir}/feature_mapping.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"\n✓ Feature mapping saved to {self.output_dir}/feature_mapping.json")
    
    def visualize_class_distribution(self, attack_dist):
        """Create visualization for attack class distribution"""
        plt.figure(figsize=(14, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        colors = sns.color_palette("husl", len(attack_dist))
        attack_dist.plot(kind='bar', color=colors, edgecolor='black')
        plt.title('Attack Type Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Attack Type', fontsize=11)
        plt.ylabel('Count', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(attack_dist.values, labels=attack_dist.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Attack Type Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/class_distribution.png", 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Class distribution plot saved")
        plt.close()
    
    def handle_missing_values(self):
        """Handle missing values intelligently"""
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)
        
        missing_before = self.df.isnull().sum().sum()
        
        if missing_before == 0:
            print("✓ No missing values to handle!")
            return True
        
        print(f"\nTotal missing values: {missing_before}")
        
        # Strategy 1: Drop columns with >50% missing
        high_missing = self.df.columns[self.df.isnull().mean() > 0.5]
        if len(high_missing) > 0:
            print(f"\nDropping {len(high_missing)} columns with >50% missing values:")
            print(f"  {list(high_missing)}")
            self.df.drop(columns=high_missing, inplace=True)
        
        # Strategy 2: Fill numeric with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Strategy 3: Fill categorical with mode
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"\n✓ Missing values after handling: {missing_after}")
        
        return True
    
    def handle_infinite_values(self):
        """Replace infinite values with appropriate substitutes"""
        print("\n" + "="*80)
        print("HANDLING INFINITE VALUES")
        print("="*80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_count = 0
        
        for col in numeric_cols:
            inf_mask = np.isinf(self.df[col])
            col_inf_count = inf_mask.sum()
            
            if col_inf_count > 0:
                inf_count += col_inf_count
                # Replace inf with max finite value * 1.5
                max_val = self.df.loc[~inf_mask, col].max()
                self.df.loc[inf_mask, col] = max_val * 1.5 if not pd.isna(max_val) else 0
        
        if inf_count > 0:
            print(f"✓ Replaced {inf_count} infinite values")
        else:
            print("✓ No infinite values detected")
        
        return True
    
    def feature_correlation_analysis(self):
        """Analyze feature correlations and identify redundant features"""
        print("\n" + "="*80)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*80)
        
        # Select numeric features only
        numeric_data = self.df[self.feature_names]
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().abs()
        
        # Find highly correlated pairs (>0.95)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (>0.95):")
            for feat1, feat2, corr_val in high_corr_pairs[:10]:
                print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
            
            # Recommend features to drop
            features_to_drop = list(set([pair[1] for pair in high_corr_pairs]))
            print(f"\n✓ Recommended features to drop: {len(features_to_drop)}")
            
            # Save correlation info
            with open(f"{self.output_dir}/high_correlations.txt", 'w') as f:
                for feat1, feat2, corr_val in high_corr_pairs:
                    f.write(f"{feat1},{feat2},{corr_val:.4f}\n")
        else:
            print("\n✓ No highly correlated feature pairs found")
        
        # Visualize correlation heatmap (top 30 features)
        if len(numeric_data.columns) > 30:
            # Select top 30 features by variance
            variances = numeric_data.var().sort_values(ascending=False)
            top_features = variances.head(30).index
            plot_data = numeric_data[top_features]
        else:
            plot_data = numeric_data
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(plot_data.corr(), cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap (Top Features)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/correlation_heatmap.png",
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Correlation heatmap saved")
        plt.close()
        
        return True
    
    def encode_labels(self):
        """Encode attack labels for classification"""
        print("\n" + "="*80)
        print("LABEL ENCODING")
        print("="*80)
        
        if not hasattr(self, 'label_col'):
            print("✗ Label column not identified!")
            return False
        
        # Encode labels
        self.df['encoded_label'] = self.label_encoder.fit_transform(self.df[self.label_col])
        
        # Create mapping
        self.attack_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        print("\nAttack Label Mapping:")
        for attack, code in sorted(self.attack_mapping.items(), key=lambda x: x[1]):
            print(f"  {code}: {attack}")
        
        # Save mapping
        # Convert numpy int64 to Python int for JSON compatibility
        attack_mapping_serializable = {str(k): int(v) for k, v in self.attack_mapping.items()}

        with open(f"{self.output_dir}/attack_mapping.json", 'w') as f:
            json.dump(attack_mapping_serializable, f, indent=2)

        
        print(f"\n✓ Label encoding complete. Mapping saved.")
        
        return True
    
    def normalize_features(self):
        """Normalize/scale numeric features"""
        print("\n" + "="*80)
        print("FEATURE NORMALIZATION")
        print("="*80)
        
        # Separate features and labels
        X = self.df[self.feature_names].values
        y = self.df['encoded_label'].values
        
        print(f"\nOriginal feature range:")
        print(f"  Min: {X.min():.2f}")
        print(f"  Max: {X.max():.2f}")
        print(f"  Mean: {X.mean():.2f}")
        
        # Apply StandardScaler
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nScaled feature range:")
        print(f"  Min: {X_scaled.min():.2f}")
        print(f"  Max: {X_scaled.max():.2f}")
        print(f"  Mean: {X_scaled.mean():.2f}")
        print(f"  Std: {X_scaled.std():.2f}")
        
        # Store scaled data back
        self.df[self.feature_names] = X_scaled
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, f"{self.output_dir}/scaler.pkl")
        print(f"\n✓ Feature normalization complete. Scaler saved.")
        
        return True
    
    def balance_dataset(self, strategy='hybrid', sampling_ratio=0.5):
        """Balance dataset using SMOTE + undersampling"""
        print("\n" + "="*80)
        print("DATASET BALANCING")
        print("="*80)
        
        X = self.df[self.feature_names].values
        y = self.df['encoded_label'].values
        
        print(f"\nOriginal class distribution:")
        original_dist = Counter(y)
        for label, count in sorted(original_dist.items()):
            attack_name = self.label_encoder.inverse_transform([label])[0]
            print(f"  {attack_name}: {count:,}")
        
        if strategy == 'smote':
            # SMOTE only
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
        elif strategy == 'undersample':
            # Random undersampling
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
            
        elif strategy == 'hybrid':
            # SMOTE + undersampling for majority class
            smote = SMOTE(random_state=42, sampling_strategy=sampling_ratio)
            X_temp, y_temp = smote.fit_resample(X, y)
            
            rus = RandomUnderSampler(random_state=42, sampling_strategy='auto')
            X_balanced, y_balanced = rus.fit_resample(X_temp, y_temp)
        
        else:
            print(f"✗ Unknown strategy: {strategy}")
            return False
        
        print(f"\nBalanced class distribution ({strategy}):")
        balanced_dist = Counter(y_balanced)
        for label, count in sorted(balanced_dist.items()):
            attack_name = self.label_encoder.inverse_transform([label])[0]
            print(f"  {attack_name}: {count:,}")
        
        # Update dataframe
        self.df = pd.DataFrame(X_balanced, columns=self.feature_names)
        self.df['encoded_label'] = y_balanced
        
        print(f"\n✓ Dataset balanced: {len(X)} → {len(X_balanced)} samples")
        
        return True
    
    def split_dataset(self, test_size=0.2, val_size=0.1):
        """Split dataset into train/validation/test sets"""
        print("\n" + "="*80)
        print("DATASET SPLITTING")
        print("="*80)
        
        X = self.df[self.feature_names].values
        y = self.df['encoded_label'].values
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"\nDataset split:")
        print(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Save splits
        np.savez(f"{self.output_dir}/dataset_splits.npz",
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test)
        
        print(f"\n✓ Dataset splits saved")
        
        return True
    
    def generate_preprocessing_report(self):
        """Generate comprehensive preprocessing report"""
        print("\n" + "="*80)
        print("GENERATING PREPROCESSING REPORT")
        print("="*80)
        
        report = {
            "dataset_info": {
                "original_shape": f"{self.df.shape[0]} samples, {self.df.shape[1]} features",
                "final_shape": f"{len(self.X_train) + len(self.X_val) + len(self.X_test)} samples",
                "feature_count": len(self.feature_names)
            },
            "preprocessing_steps": [
                "Missing value handling",
                "Infinite value replacement",
                "Feature correlation analysis",
                "Label encoding",
                "Feature normalization (StandardScaler)",
                "Dataset balancing (SMOTE + Undersampling)",
                "Train/Val/Test split"
            ],
            "attack_types": list(self.attack_mapping.keys()),
            "features": self.feature_names,
            "splits": {
                "train": int(len(self.X_train)),
                "validation": int(len(self.X_val)),
                "test": int(len(self.X_test))
            }
        }
        
        with open(f"{self.output_dir}/preprocessing_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Preprocessing report saved to {self.output_dir}/preprocessing_report.json")
        print("\n" + "="*80)
        print("PHASE 1 COMPLETE ✓")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review visualizations in processed_data/visualizations/")
        print("  2. Check feature_mapping.json for real-time capture compatibility")
        print("  3. Proceed to Phase 2: Model Building & Optimization")
        
        return True
    
    def run_full_pipeline(self, balance_strategy='hybrid', test_size=0.2):
        """Execute complete preprocessing pipeline"""
        steps = [
            ("Loading Data", self.load_data),
            ("Exploring Data", self.explore_data),
            ("Handling Missing Values", self.handle_missing_values),
            ("Handling Infinite Values", self.handle_infinite_values),
            ("Correlation Analysis", self.feature_correlation_analysis),
            ("Encoding Labels", self.encode_labels),
            ("Normalizing Features", self.normalize_features),
            ("Balancing Dataset", lambda: self.balance_dataset(strategy=balance_strategy)),
            ("Splitting Dataset", lambda: self.split_dataset(test_size=test_size)),
            ("Generating Report", self.generate_preprocessing_report)
        ]
        
        for step_name, step_func in steps:
            try:
                result = step_func()
                if not result:
                    print(f"\n✗ Step '{step_name}' failed!")
                    return False
            except Exception as e:
                print(f"\n✗ Error in step '{step_name}': {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TIIDataPreprocessor(
        data_path='TII-SSRC-23/csv/data.csv',
        output_dir='processed_data'
    )
    
    # Run full pipeline
    success = preprocessor.run_full_pipeline(
        balance_strategy='hybrid',  # Options: 'smote', 'undersample', 'hybrid'
        test_size=0.2
    )
    
    if success:
        print("\n🎉 Preprocessing completed successfully!")
        print("\nSaved artifacts:")
        print("  • processed_data/dataset_splits.npz")
        print("  • processed_data/scaler.pkl")
        print("  • processed_data/attack_mapping.json")
        print("  • processed_data/feature_mapping.json")
        print("  • processed_data/preprocessing_report.json")
        print("  • processed_data/visualizations/")
    else:
        print("\n❌ Preprocessing failed. Check error messages above.")