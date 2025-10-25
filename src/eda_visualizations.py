"""
SentiNet - Advanced Exploratory Data Analysis
Comprehensive visualization suite for TII-SSRC-23 dataset

This module provides in-depth visual analysis to understand:
- Attack patterns and distributions
- Feature importance and relationships
- Data quality and anomalies
- Temporal patterns (if timestamps available)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import json
import os

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedEDA:
    """Advanced EDA and visualization suite"""
    
    def __init__(self, data_path='processed_data/dataset_splits.npz',
                 mapping_path='processed_data/attack_mapping.json',
                 feature_path='processed_data/feature_mapping.json',
                 output_dir='processed_data/visualizations'):
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load preprocessed data
        data = np.load(data_path)
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        
        # Load mappings
        with open(mapping_path, 'r') as f:
            self.attack_mapping = json.load(f)
        
        with open(feature_path, 'r') as f:
            feature_info = json.load(f)
            self.feature_names = feature_info['all_features']
        
        # Reverse mapping for labels
        self.label_to_attack = {v: k for k, v in self.attack_mapping.items()}
        
        print(f"✓ EDA initialized")
        print(f"  Training samples: {len(self.X_train):,}")
        print(f"  Test samples: {len(self.X_test):,}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Attack types: {len(self.attack_mapping)}")
    
    def plot_feature_distributions(self, n_features=16):
        """Plot distribution of top features"""
        print("\n" + "="*80)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("="*80)
        
        # Select features with highest variance
        variances = np.var(self.X_train, axis=0)
        top_indices = np.argsort(variances)[-n_features:]
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            
            feature_data = self.X_train[:, feat_idx]
            feature_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
            
            ax.hist(feature_data, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{feature_name}\n(var={variances[feat_idx]:.3f})', fontsize=9)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Distribution of Top 16 Features (by Variance)', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_distributions.png", 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Feature distributions saved")
        plt.close()
    
    def plot_attack_wise_features(self, n_features=6):
        """Plot feature distributions grouped by attack type"""
        print("\nGenerating attack-wise feature analysis...")
        
        # Select most discriminative features
        variances = np.var(self.X_train, axis=0)
        top_indices = np.argsort(variances)[-n_features:]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            feature_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
            
            # Create dataframe for visualization
            plot_data = []
            for attack_label in np.unique(self.y_train):
                attack_name = self.label_to_attack.get(attack_label, f"Class_{attack_label}")
                mask = self.y_train == attack_label
                feature_values = self.X_train[mask, feat_idx]
                
                # Sample if too many points
                if len(feature_values) > 1000:
                    feature_values = np.random.choice(feature_values, 1000, replace=False)
                
                for val in feature_values:
                    plot_data.append({'Attack': attack_name, 'Value': val})
            
            df_plot = pd.DataFrame(plot_data)
            
            # Violin plot
            sns.violinplot(data=df_plot, x='Attack', y='Value', ax=ax, cut=0)
            ax.set_title(feature_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Normalized Value', fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Feature Distributions by Attack Type', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/attack_wise_features.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Attack-wise feature analysis saved")
        plt.close()
    
    def plot_pca_analysis(self, n_components=2):
        """Perform and visualize PCA dimensionality reduction"""
        print("\nPerforming PCA analysis...")
        
        # Sample data if too large
        if len(self.X_train) > 10000:
            indices = np.random.choice(len(self.X_train), 10000, replace=False)
            X_sample = self.X_train[indices]
            y_sample = self.y_train[indices]
        else:
            X_sample = self.X_train
            y_sample = self.y_train
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_sample)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        ax1 = axes[0]
        for attack_label in np.unique(y_sample):
            attack_name = self.label_to_attack.get(attack_label, f"Class_{attack_label}")
            mask = y_sample == attack_label
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=attack_name, alpha=0.6, s=20)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                      fontsize=11)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                      fontsize=11)
        ax1.set_title('PCA: Attack Type Clustering', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Explained variance
        ax2 = axes[1]
        pca_full = PCA()
        pca_full.fit(X_sample)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        
        ax2.plot(range(1, len(cumsum[:50])+1), cumsum[:50], 'b-o', markersize=4)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax2.axhline(y=0.99, color='orange', linestyle='--', label='99% Variance')
        ax2.set_xlabel('Number of Components', fontsize=11)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=11)
        ax2.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Find components needed for 95% variance
        n_95 = np.argmax(cumsum >= 0.95) + 1
        ax2.annotate(f'{n_95} components\nfor 95% variance',
                    xy=(n_95, 0.95), xytext=(n_95+10, 0.85),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pca_analysis.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ PCA analysis saved")
        print(f"  - {n_95} components explain 95% of variance")
        print(f"  - Total variance explained by PC1+PC2: {sum(pca.explained_variance_ratio_)*100:.2f}%")
        plt.close()
    
    def plot_tsne_visualization(self, perplexity=30, n_samples=5000):
        """Create t-SNE visualization"""
        print("\nPerforming t-SNE visualization (this may take a while)...")
        
        # Sample data
        if len(self.X_train) > n_samples:
            indices = np.random.choice(len(self.X_train), n_samples, replace=False)
            X_sample = self.X_train[indices]
            y_sample = self.y_train[indices]
        else:
            X_sample = self.X_train
            y_sample = self.y_train
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                   n_iter=1000, verbose=0)
        X_tsne = tsne.fit_transform(X_sample)
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        for attack_label in np.unique(y_sample):
            attack_name = self.label_to_attack.get(attack_label, f"Class_{attack_label}")
            mask = y_sample == attack_label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       label=attack_name, alpha=0.6, s=30)
        
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE Visualization of Attack Types', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/tsne_visualization.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ t-SNE visualization saved")
        plt.close()
    
    def plot_feature_importance_proxy(self, n_top=20):
        """Calculate and plot feature importance using variance and separability"""
        print("\nCalculating feature importance metrics...")
        
        # Method 1: Variance-based importance
        variances = np.var(self.X_train, axis=0)
        
        # Method 2: Class separability (simplified)
        separability_scores = []
        for feat_idx in range(self.X_train.shape[1]):
            class_means = []
            for attack_label in np.unique(self.y_train):
                mask = self.y_train == attack_label
                class_means.append(np.mean(self.X_train[mask, feat_idx]))
            
            # Standard deviation of class means (higher = more separable)
            separability_scores.append(np.std(class_means))
        
        separability_scores = np.array(separability_scores)
        
        # Combined score
        # Normalize both metrics
        variances_norm = (variances - variances.min()) / (variances.max() - variances.min() + 1e-10)
        separability_norm = (separability_scores - separability_scores.min()) / (separability_scores.max() - separability_scores.min() + 1e-10)
        
        combined_scores = 0.5 * variances_norm + 0.5 * separability_norm
        
        # Get top features
        top_indices = np.argsort(combined_scores)[-n_top:][::-1]
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Variance importance
        ax1 = axes[0]
        top_var_indices = np.argsort(variances)[-n_top:][::-1]
        feature_names_var = [self.feature_names[i] if i < len(self.feature_names) else f"F{i}" 
                            for i in top_var_indices]
        ax1.barh(range(n_top), variances[top_var_indices], color='steelblue', edgecolor='black')
        ax1.set_yticks(range(n_top))
        ax1.set_yticklabels(feature_names_var, fontsize=8)
        ax1.set_xlabel('Variance', fontsize=11)
        ax1.set_title('Top Features by Variance', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Separability importance
        ax2 = axes[1]
        top_sep_indices = np.argsort(separability_scores)[-n_top:][::-1]
        feature_names_sep = [self.feature_names[i] if i < len(self.feature_names) else f"F{i}"
                            for i in top_sep_indices]
        ax2.barh(range(n_top), separability_scores[top_sep_indices], 
                color='coral', edgecolor='black')
        ax2.set_yticks(range(n_top))
        ax2.set_yticklabels(feature_names_sep, fontsize=8)
        ax2.set_xlabel('Class Separability Score', fontsize=11)
        ax2.set_title('Top Features by Separability', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Combined importance
        ax3 = axes[2]
        feature_names_combined = [self.feature_names[i] if i < len(self.feature_names) else f"F{i}"
                                 for i in top_indices]
        ax3.barh(range(n_top), combined_scores[top_indices], 
                color='mediumseagreen', edgecolor='black')
        ax3.set_yticks(range(n_top))
        ax3.set_yticklabels(feature_names_combined, fontsize=8)
        ax3.set_xlabel('Combined Importance Score', fontsize=11)
        ax3.set_title('Top Features (Combined Score)', fontsize=12, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance analysis saved")
        
        # Save top features to JSON
        top_features_info = {
            'top_variance': [self.feature_names[i] if i < len(self.feature_names) else f"F{i}"
                           for i in top_var_indices[:10]],
            'top_separability': [self.feature_names[i] if i < len(self.feature_names) else f"F{i}"
                                for i in top_sep_indices[:10]],
            'top_combined': [self.feature_names[i] if i < len(self.feature_names) else f"F{i}"
                           for i in top_indices[:10]]
        }
        
        with open(f"{self.output_dir}/../top_features.json", 'w') as f:
            json.dump(top_features_info, f, indent=2)
        
        plt.close()
    
    def plot_class_balance(self):
        """Visualize class balance in train and test sets"""
        print("\nAnalyzing class balance...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Training set bar chart
        ax1 = axes[0, 0]
        train_counts = np.bincount(self.y_train)
        train_labels = [self.label_to_attack.get(i, f"Class_{i}") 
                       for i in range(len(train_counts))]
        colors = sns.color_palette("husl", len(train_counts))
        
        ax1.bar(train_labels, train_counts, color=colors, edgecolor='black')
        ax1.set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Attack Type', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, (label, count) in enumerate(zip(train_labels, train_counts)):
            ax1.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Test set bar chart
        ax2 = axes[0, 1]
        test_counts = np.bincount(self.y_test)
        test_labels = [self.label_to_attack.get(i, f"Class_{i}") 
                      for i in range(len(test_counts))]
        
        ax2.bar(test_labels, test_counts, color=colors, edgecolor='black')
        ax2.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Attack Type', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (label, count) in enumerate(zip(test_labels, test_counts)):
            ax2.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Training set pie chart
        ax3 = axes[1, 0]
        ax3.pie(train_counts, labels=train_labels, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax3.set_title('Training Set Percentage', fontsize=12, fontweight='bold')
        
        # Test set pie chart
        ax4 = axes[1, 1]
        ax4.pie(test_counts, labels=test_labels, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax4.set_title('Test Set Percentage', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/class_balance_detailed.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Class balance visualization saved")
        plt.close()
    
    def plot_data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("\nGenerating data quality report...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Feature value ranges
        ax1 = fig.add_subplot(gs[0, :2])
        feature_mins = np.min(self.X_train, axis=0)
        feature_maxs = np.max(self.X_train, axis=0)
        feature_ranges = feature_maxs - feature_mins
        
        # Sample top features by range
        top_range_indices = np.argsort(feature_ranges)[-30:]
        
        ax1.scatter(top_range_indices, feature_mins[top_range_indices], 
                   label='Min', alpha=0.6, s=50)
        ax1.scatter(top_range_indices, feature_maxs[top_range_indices], 
                   label='Max', alpha=0.6, s=50)
        ax1.fill_between(top_range_indices, 
                        feature_mins[top_range_indices],
                        feature_maxs[top_range_indices],
                        alpha=0.2)
        ax1.set_xlabel('Feature Index', fontsize=10)
        ax1.set_ylabel('Normalized Value', fontsize=10)
        ax1.set_title('Feature Value Ranges (Top 30 by Range)', 
                     fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Outlier detection (IQR method)
        ax2 = fig.add_subplot(gs[0, 2])
        outlier_counts = []
        for feat_idx in range(self.X_train.shape[1]):
            Q1 = np.percentile(self.X_train[:, feat_idx], 25)
            Q3 = np.percentile(self.X_train[:, feat_idx], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.sum((self.X_train[:, feat_idx] < lower_bound) | 
                            (self.X_train[:, feat_idx] > upper_bound))
            outlier_counts.append(outliers)
        
        outlier_pcts = (np.array(outlier_counts) / len(self.X_train)) * 100
        
        ax2.hist(outlier_pcts, bins=30, color='salmon', edgecolor='black')
        ax2.set_xlabel('Outlier %', fontsize=10)
        ax2.set_ylabel('Feature Count', fontsize=10)
        ax2.set_title('Outlier Distribution\n(IQR Method)', 
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature sparsity
        ax3 = fig.add_subplot(gs[1, 0])
        zero_counts = np.sum(self.X_train == 0, axis=0)
        sparsity = (zero_counts / len(self.X_train)) * 100
        
        ax3.hist(sparsity, bins=30, color='lightblue', edgecolor='black')
        ax3.set_xlabel('Sparsity (%)', fontsize=10)
        ax3.set_ylabel('Feature Count', fontsize=10)
        ax3.set_title('Feature Sparsity Distribution', 
                     fontsize=11, fontweight='bold')
        ax3.axvline(x=50, color='r', linestyle='--', label='50% threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature variance
        ax4 = fig.add_subplot(gs[1, 1])
        variances = np.var(self.X_train, axis=0)
        
        ax4.hist(np.log10(variances + 1e-10), bins=30, 
                color='lightgreen', edgecolor='black')
        ax4.set_xlabel('Log10(Variance)', fontsize=10)
        ax4.set_ylabel('Feature Count', fontsize=10)
        ax4.set_title('Feature Variance Distribution', 
                     fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Sample distribution by class
        ax5 = fig.add_subplot(gs[1, 2])
        class_counts = np.bincount(self.y_train)
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        ax5.bar(range(len(class_counts)), class_counts, 
               color='mediumpurple', edgecolor='black')
        ax5.set_xlabel('Class Label', fontsize=10)
        ax5.set_ylabel('Sample Count', fontsize=10)
        ax5.set_title(f'Class Distribution\n(Imbalance Ratio: {imbalance_ratio:.2f})', 
                     fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Feature correlation summary
        ax6 = fig.add_subplot(gs[2, 0])
        # Sample features for correlation calculation
        sample_size = min(50, self.X_train.shape[1])
        sample_indices = np.random.choice(self.X_train.shape[1], sample_size, replace=False)
        corr_matrix = np.corrcoef(self.X_train[:, sample_indices].T)
        
        # Get upper triangle correlations
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        ax6.hist(upper_triangle, bins=30, color='orange', edgecolor='black')
        ax6.set_xlabel('Correlation Coefficient', fontsize=10)
        ax6.set_ylabel('Frequency', fontsize=10)
        ax6.set_title('Feature Correlation Distribution\n(Sample of 50 features)', 
                     fontsize=11, fontweight='bold')
        ax6.axvline(x=0.9, color='r', linestyle='--', label='High correlation')
        ax6.axvline(x=-0.9, color='r', linestyle='--')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Summary statistics table
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('tight')
        ax7.axis('off')
        
        summary_stats = [
            ['Metric', 'Value'],
            ['Total Features', f'{self.X_train.shape[1]}'],
            ['Training Samples', f'{len(self.X_train):,}'],
            ['Test Samples', f'{len(self.X_test):,}'],
            ['Attack Classes', f'{len(np.unique(self.y_train))}'],
            ['Avg Feature Variance', f'{np.mean(variances):.4f}'],
            ['Features >50% Sparse', f'{np.sum(sparsity > 50)}'],
            ['High Correlation Pairs', f'{np.sum(np.abs(upper_triangle) > 0.9)}'],
            ['Class Imbalance Ratio', f'{imbalance_ratio:.2f}']
        ]
        
        table = ax7.table(cellText=summary_stats, cellLoc='left',
                         colWidths=[0.5, 0.5], loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_stats)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax7.set_title('Data Quality Summary Statistics', 
                     fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Comprehensive Data Quality Report', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.savefig(f"{self.output_dir}/data_quality_report.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Data quality report saved")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization reports"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE VISUALIZATION SUITE")
        print("="*80)
        
        visualizations = [
            ("Feature Distributions", self.plot_feature_distributions),
            ("Attack-wise Features", self.plot_attack_wise_features),
            ("PCA Analysis", self.plot_pca_analysis),
            ("t-SNE Visualization", self.plot_tsne_visualization),
            ("Feature Importance", self.plot_feature_importance_proxy),
            ("Class Balance", self.plot_class_balance),
            ("Data Quality Report", self.plot_data_quality_report)
        ]
        
        for viz_name, viz_func in visualizations:
            try:
                viz_func()
            except Exception as e:
                print(f"\n✗ Error generating {viz_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*80)
        print("VISUALIZATION SUITE COMPLETE ✓")
        print("="*80)
        print(f"\nAll visualizations saved to: {self.output_dir}/")
        print("\nGenerated files:")
        print("  • feature_distributions.png")
        print("  • attack_wise_features.png")
        print("  • pca_analysis.png")
        print("  • tsne_visualization.png")
        print("  • feature_importance.png")
        print("  • class_balance_detailed.png")
        print("  • data_quality_report.png")


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("SENTINET - ADVANCED EDA & VISUALIZATION")
    print("="*80)
    
    try:
        # Initialize EDA
        eda = AdvancedEDA(
            data_path='processed_data/dataset_splits.npz',
            mapping_path='processed_data/attack_mapping.json',
            feature_path='processed_data/feature_mapping.json',
            output_dir='processed_data/visualizations'
        )
        
        # Generate all visualizations
        eda.generate_all_visualizations()
        
        print("\n✅ EDA and visualization complete!")
        print("\nNext steps:")
        print("  1. Review all visualizations in processed_data/visualizations/")
        print("  2. Analyze feature importance for feature selection")
        print("  3. Proceed to Phase 2: Model Training")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required files not found")
        print(f"   {e}")
        print("\n⚠ Please run preprocessing.py first to generate required files")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()