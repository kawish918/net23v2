"""
SentiNet - Phase 2: Model Building & Optimization
Advanced Multi-Model Training Pipeline with Hyperparameter Optimization

This module implements state-of-the-art machine learning models for
intrusion detection with emphasis on:
- High accuracy and F1-scores
- Fast inference speed for real-time deployment
- Robust cross-validation
- Comprehensive performance metrics
- Model versioning and export

Author: SentiNet Development Team
Version: 2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Comprehensive model training and evaluation system
    Trains multiple models, performs hyperparameter tuning, and selects best model
    """
    
    def __init__(self, data_dir='processed_data', models_dir='models', results_dir='results'):
        """
        Initialize model trainer
        
        Args:
            data_dir: Directory containing preprocessed data
            models_dir: Directory to save trained models
            results_dir: Directory to save results and metrics
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        
        # Load preprocessed data
        print("="*80)
        print("INITIALIZING MODEL TRAINER")
        print("="*80)
        
        self._load_data()
        self._load_mappings()
        
        # Initialize model registry
        self.trained_models = {}
        self.performance_metrics = {}
        self.training_times = {}
        self.inference_times = {}
        
        print(f"\n✓ Trainer initialized successfully")
        print(f"  - Training samples: {len(self.X_train):,}")
        print(f"  - Validation samples: {len(self.X_val):,}")
        print(f"  - Test samples: {len(self.X_test):,}")
        print(f"  - Features: {self.X_train.shape[1]}")
        print(f"  - Classes: {len(self.label_to_attack)}")
    
    def _load_data(self):
        """Load preprocessed dataset splits"""
        data_path = f"{self.data_dir}/dataset_splits.npz"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset splits not found at {data_path}\n"
                "Please run preprocessing.py first (Phase 1)"
            )
        
        data = np.load(data_path)
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        
        print(f"✓ Dataset loaded from {data_path}")
    
    def _load_mappings(self):
        """Load label mappings and feature info"""
        # Load attack mapping
        mapping_path = f"{self.data_dir}/attack_mapping.json"
        with open(mapping_path, 'r') as f:
            self.attack_mapping = json.load(f)
        
        self.label_to_attack = {v: k for k, v in self.attack_mapping.items()}
        
        # Load feature names
        feature_path = f"{self.data_dir}/feature_mapping.json"
        with open(feature_path, 'r') as f:
            feature_info = json.load(f)
            self.feature_names = feature_info['all_features']
        
        print(f"✓ Mappings loaded")
    
    def train_random_forest(self, **kwargs):
        """
        Train Random Forest classifier with optimized hyperparameters
        
        Args:
            **kwargs: Custom hyperparameters for RandomForestClassifier
        
        Returns:
            Trained model
        """
        print("\n" + "="*80)
        print("TRAINING: RANDOM FOREST CLASSIFIER")
        print("="*80)
        
        # Optimized default parameters (reduced for memory efficiency with large datasets)
        default_params = {
            'n_estimators': 100,  # Reduced from 200
            'max_depth': 20,      # Reduced from 30
            'min_samples_split': 10,  # Increased for memory
            'min_samples_leaf': 5,    # Increased for memory
            'max_features': 'sqrt',
            'n_jobs': 10,         # Reduced from -1 (use half cores)
            'random_state': 42,
            'class_weight': 'balanced',
            'verbose': 1,
            'max_samples': 0.7    # Use 70% of data per tree
        }
        
        # Update with custom parameters
        params = {**default_params, **kwargs}
        
        print(f"\nHyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Train model
        start_time = time.time()
        
        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # CHECKPOINT: Save model immediately after training
        checkpoint_path = f"{self.models_dir}/checkpoint_RandomForest.pkl"
        joblib.dump(model, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Evaluate
        metrics = self._evaluate_model(model, 'RandomForest')
        
        # Store
        self.trained_models['RandomForest'] = model
        self.performance_metrics['RandomForest'] = metrics
        self.training_times['RandomForest'] = training_time
        
        return model
    
    def train_xgboost(self, **kwargs):
        """
        Train XGBoost classifier with optimized hyperparameters
        
        Args:
            **kwargs: Custom hyperparameters for XGBClassifier
        
        Returns:
            Trained model
        """
        print("\n" + "="*80)
        print("TRAINING: XGBOOST CLASSIFIER")
        print("="*80)
        
        # Calculate scale_pos_weight for class imbalance
        class_counts = Counter(self.y_train)
        majority_class = max(class_counts.values())
        scale_weights = {k: majority_class/v for k, v in class_counts.items()}
        
        # Optimized default parameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',  # Fixed for binary classification
            'eval_metric': 'logloss',        # Fixed for binary
            'n_jobs': 10,                    # Reduced for memory
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 1
        }
        
        # Update with custom parameters
        params = {**default_params, **kwargs}
        
        print(f"\nHyperparameters:")
        for key, value in params.items():
            if key != 'sample_weight':
                print(f"  {key}: {value}")
        
        # Train model
        start_time = time.time()
        
        model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        eval_set = [(self.X_val, self.y_val)]
        model.fit(
            self.X_train, self.y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        print(f"  Best iteration: {model.best_iteration}")
        
        # CHECKPOINT: Save model immediately after training
        checkpoint_path = f"{self.models_dir}/checkpoint_XGBoost.pkl"
        joblib.dump(model, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Evaluate
        metrics = self._evaluate_model(model, 'XGBoost')
        
        # Store
        self.trained_models['XGBoost'] = model
        self.performance_metrics['XGBoost'] = metrics
        self.training_times['XGBoost'] = training_time
        
        return model
    
    def train_lightgbm(self, **kwargs):
        """
        Train LightGBM classifier with optimized hyperparameters
        
        Args:
            **kwargs: Custom hyperparameters for LGBMClassifier
        
        Returns:
            Trained model
        """
        print("\n" + "="*80)
        print("TRAINING: LIGHTGBM CLASSIFIER")
        print("="*80)
        
        # Optimized default parameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'objective': 'binary',           # Fixed for binary classification
            'metric': 'binary_logloss',      # Fixed for binary
            'n_jobs': 10,                    # Reduced for memory
            'random_state': 42,
            'class_weight': 'balanced',
            'verbose': -1
        }
        
        # Update with custom parameters
        params = {**default_params, **kwargs}
        
        print(f"\nHyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Train model
        start_time = time.time()
        
        model = lgb.LGBMClassifier(**params)
        
        # Train with early stopping
        eval_set = [(self.X_val, self.y_val)]
        model.fit(
            self.X_train, self.y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        print(f"  Best iteration: {model.best_iteration_}")
        
        # CHECKPOINT: Save model immediately after training
        checkpoint_path = f"{self.models_dir}/checkpoint_LightGBM.pkl"
        joblib.dump(model, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Evaluate
        metrics = self._evaluate_model(model, 'LightGBM')
        
        # Store
        self.trained_models['LightGBM'] = model
        self.performance_metrics['LightGBM'] = metrics
        self.training_times['LightGBM'] = training_time
        
        return model
    
    def train_gradient_boosting(self, **kwargs):
        """
        Train Gradient Boosting classifier
        
        Args:
            **kwargs: Custom hyperparameters for GradientBoostingClassifier
        
        Returns:
            Trained model
        """
        print("\n" + "="*80)
        print("TRAINING: GRADIENT BOOSTING CLASSIFIER")
        print("="*80)
        
        # Optimized default parameters (memory efficient)
        default_params = {
            'n_estimators': 100,  # Reduced from 150 for memory
            'max_depth': 7,       # Reduced for memory
            'learning_rate': 0.1,
            'subsample': 0.8,
            'min_samples_split': 10,  # Increased for memory
            'min_samples_leaf': 5,    # Increased for memory
            'max_features': 'sqrt',
            'random_state': 42,
            'verbose': 1
        }
        
        # Update with custom parameters
        params = {**default_params, **kwargs}
        
        print(f"\nHyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Train model
        start_time = time.time()
        
        model = GradientBoostingClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # CHECKPOINT: Save model immediately after training
        checkpoint_path = f"{self.models_dir}/checkpoint_GradientBoosting.pkl"
        joblib.dump(model, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Evaluate
        metrics = self._evaluate_model(model, 'GradientBoosting')
        
        # Store
        self.trained_models['GradientBoosting'] = model
        self.performance_metrics['GradientBoosting'] = metrics
        self.training_times['GradientBoosting'] = training_time
        
        return model
    
    def _evaluate_model(self, model, model_name):
        """
        Comprehensive model evaluation on train, validation, and test sets
        
        Args:
            model: Trained model
            model_name: Name identifier for the model
        
        Returns:
            Dictionary of performance metrics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*80}")
        
        metrics = {}
        
        # Evaluate on each dataset split
        for split_name, X, y in [
            ('train', self.X_train, self.y_train),
            ('val', self.X_val, self.y_val),
            ('test', self.X_test, self.y_test)
        ]:
            # Predictions
            start_time = time.time()
            y_pred = model.predict(X)
            inference_time = (time.time() - start_time) / len(X)  # Per sample
            
            # Probabilities for multi-class ROC-AUC
            try:
                y_proba = model.predict_proba(X)
            except:
                y_proba = None
            
            # Calculate metrics
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            # ROC-AUC calculation
            try:
                if y_proba is not None:
                    # For binary classification, use probability of positive class
                    if y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y, y_proba[:, 1])
                    else:
                        # Multi-class
                        roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
                else:
                    roc_auc = None
            except:
                roc_auc = None
            
            metrics[split_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'inference_time_per_sample': inference_time
            }
            
            print(f"\n{split_name.upper()} SET PERFORMANCE:")
            print(f"  Accuracy:  {acc*100:.2f}%")
            print(f"  Precision: {prec*100:.2f}%")
            print(f"  Recall:    {rec*100:.2f}%")
            print(f"  F1-Score:  {f1*100:.2f}%")
            if roc_auc:
                print(f"  ROC-AUC:   {roc_auc:.4f}")
            print(f"  Inference: {inference_time*1000:.3f} ms/sample")
        
        # Store inference time
        self.inference_times[model_name] = metrics['test']['inference_time_per_sample']
        
        # Generate confusion matrix and classification report for test set
        self._generate_confusion_matrix(model, model_name)
        self._generate_classification_report(model, model_name)
        
        return metrics
    
    def _generate_confusion_matrix(self, model, model_name):
        """Generate and save confusion matrix visualization"""
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Raw counts
        ax1 = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=[self.label_to_attack[i] for i in range(len(cm))],
                   yticklabels=[self.label_to_attack[i] for i in range(len(cm))],
                   cbar_kws={'label': 'Count'})
        ax1.set_title(f'{model_name} - Confusion Matrix (Counts)', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=11)
        ax1.set_xlabel('Predicted Label', fontsize=11)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)
        
        # Normalized
        ax2 = axes[1]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                   xticklabels=[self.label_to_attack[i] for i in range(len(cm))],
                   yticklabels=[self.label_to_attack[i] for i in range(len(cm))],
                   cbar_kws={'label': 'Percentage'})
        ax2.set_title(f'{model_name} - Confusion Matrix (Normalized)',
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=11)
        ax2.set_xlabel('Predicted Label', fontsize=11)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/{model_name}_confusion_matrix.png",
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved")
        plt.close()
    
    def _generate_classification_report(self, model, model_name):
        """Generate and save detailed classification report"""
        y_pred = model.predict(self.X_test)
        
        # Get classification report as dict
        report_dict = classification_report(
            self.y_test, y_pred,
            target_names=[self.label_to_attack[i] for i in sorted(self.label_to_attack.keys())],
            output_dict=True,
            zero_division=0
        )
        
        # Save as JSON
        report_path = f"{self.results_dir}/{model_name}_classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Also save as text
        report_text = classification_report(
            self.y_test, y_pred,
            target_names=[self.label_to_attack[i] for i in sorted(self.label_to_attack.keys())],
            zero_division=0
        )
        
        text_path = f"{self.results_dir}/{model_name}_classification_report.txt"
        with open(text_path, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Classification report saved")
        print(f"\nPer-Class Performance (F1-Scores):")
        for class_name, metrics in report_dict.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                f1 = metrics.get('f1-score', 0)
                print(f"  {class_name}: {f1*100:.2f}%")
    
    def perform_cross_validation(self, model, model_name, cv_folds=5):
        """
        Perform k-fold cross-validation
        
        Args:
            model: Model to validate
            model_name: Model identifier
            cv_folds: Number of CV folds
        
        Returns:
            Cross-validation scores
        """
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION: {model_name} ({cv_folds}-Fold)")
        print(f"{'='*80}")
        
        # Combine train and validation for CV
        X_combined = np.vstack([self.X_train, self.X_val])
        y_combined = np.concatenate([self.y_train, self.y_val])
        
        # Perform CV
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(
            model, X_combined, y_combined,
            cv=skf, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        print(f"\nCross-Validation Results:")
        print(f"  F1-Scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"  Mean F1:   {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
        
        # Store CV results
        if 'cross_validation' not in self.performance_metrics.get(model_name, {}):
            if model_name not in self.performance_metrics:
                self.performance_metrics[model_name] = {}
            self.performance_metrics[model_name]['cross_validation'] = {
                'scores': cv_scores.tolist(),
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
        
        return cv_scores
    
    def plot_feature_importance(self, model, model_name, top_n=20):
        """
        Plot and save feature importance
        
        Args:
            model: Trained model with feature_importances_ attribute
            model_name: Model identifier
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"\n⚠ {model_name} does not support feature importance")
            return
        
        print(f"\nGenerating feature importance plot for {model_name}...")
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        feature_names_top = [self.feature_names[i] if i < len(self.feature_names) 
                            else f"Feature_{i}" for i in indices]
        
        plt.barh(range(top_n), importances[indices], color='steelblue', edgecolor='black')
        plt.yticks(range(top_n), feature_names_top, fontsize=9)
        plt.xlabel('Importance Score', fontsize=11)
        plt.title(f'{model_name} - Top {top_n} Feature Importances',
                 fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plt.savefig(f"{self.results_dir}/plots/{model_name}_feature_importance.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved")
        plt.close()
        
        # Save top features to JSON
        top_features = {
            self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}": 
            float(importances[i])
            for i in indices
        }
        
        with open(f"{self.results_dir}/{model_name}_top_features.json", 'w') as f:
            json.dump(top_features, f, indent=2)
    
    def train_all_models(self, perform_cv=True):
        """
        Train all available models with checkpoint recovery
        
        Args:
            perform_cv: Whether to perform cross-validation
        
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*80)
        print("TRAINING ALL MODELS WITH CHECKPOINT SYSTEM")
        print("="*80)
        
        # Check for existing checkpoints
        checkpoint_info = self._check_existing_checkpoints()
        if checkpoint_info:
            print("\n⚠️  CHECKPOINT RECOVERY AVAILABLE")
            print("Found previously saved checkpoints:")
            for model_name in checkpoint_info:
                print(f"  ✓ {model_name}")
            print("\nYou can load these to skip re-training if needed.")
            print("Continuing with fresh training...\n")
        
        models_to_train = [
            ('RandomForest', self.train_random_forest),
            ('XGBoost', self.train_xgboost),
            ('LightGBM', self.train_lightgbm),
            ('GradientBoosting', self.train_gradient_boosting)
        ]
        
        total_models = len(models_to_train)
        
        for idx, (model_name, train_func) in enumerate(models_to_train, 1):
            try:
                print(f"\n{'='*80}")
                print(f"PROGRESS: MODEL {idx}/{total_models} - {model_name}")
                print(f"{'='*80}")
                
                model = train_func()
                
                # Cross-validation
                if perform_cv:
                    self.perform_cross_validation(model, model_name, cv_folds=5)
                
                # Feature importance
                self.plot_feature_importance(model, model_name, top_n=20)
                
                # CHECKPOINT: Save progress after each model completes
                self._save_progress_checkpoint(idx, total_models)
                
                print(f"\n✅ {model_name} complete ({idx}/{total_models})")
                
            except Exception as e:
                print(f"\n✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n⚠️  Continuing with remaining models...")
        
        print(f"\n{'='*80}")
        print(f"✅ ALL MODELS TRAINING COMPLETE!")
        print(f"{'='*80}")
        
        return self.trained_models
    
    def _check_existing_checkpoints(self):
        """Check for existing checkpoint files"""
        checkpoint_models = []
        checkpoint_files = [
            'checkpoint_RandomForest.pkl',
            'checkpoint_XGBoost.pkl',
            'checkpoint_LightGBM.pkl',
            'checkpoint_GradientBoosting.pkl'
        ]
        
        for checkpoint_file in checkpoint_files:
            checkpoint_path = f"{self.models_dir}/{checkpoint_file}"
            if os.path.exists(checkpoint_path):
                model_name = checkpoint_file.replace('checkpoint_', '').replace('.pkl', '')
                checkpoint_models.append(model_name)
        
        return checkpoint_models
    
    def _save_progress_checkpoint(self, current_model, total_models):
        """Save training progress to file"""
        progress = {
            'completed_models': current_model,
            'total_models': total_models,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'trained_models': list(self.trained_models.keys())
        }
        
        progress_path = f"{self.results_dir}/training_progress.json"
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"  💾 Progress saved: {current_model}/{total_models} models complete")
    
    def compare_models(self):
        """Generate comprehensive model comparison report"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        if not self.trained_models:
            print("No models trained yet!")
            return
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name in self.trained_models.keys():
            metrics = self.performance_metrics[model_name]
            
            row = {
                'Model': model_name,
                'Test Accuracy': metrics['test']['accuracy'] * 100,
                'Test Precision': metrics['test']['precision'] * 100,
                'Test Recall': metrics['test']['recall'] * 100,
                'Test F1-Score': metrics['test']['f1_score'] * 100,
                'Training Time (s)': self.training_times[model_name],
                'Inference (ms)': self.inference_times[model_name] * 1000
            }
            
            if 'cross_validation' in metrics:
                row['CV F1-Score'] = metrics['cross_validation']['mean'] * 100
            
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Test F1-Score', ascending=False)
        
        # Print comparison table
        print("\n" + df_comparison.to_string(index=False))
        
        # Save to CSV
        df_comparison.to_csv(f"{self.results_dir}/model_comparison.csv", index=False)
        print(f"\n✓ Comparison table saved to {self.results_dir}/model_comparison.csv")
        
        # Visualize comparison
        self._plot_model_comparison(df_comparison)
        
        # Identify best model
        best_model_name = df_comparison.iloc[0]['Model']
        best_f1 = df_comparison.iloc[0]['Test F1-Score']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   F1-Score: {best_f1:.2f}%")
        
        return df_comparison
    
    def _plot_model_comparison(self, df_comparison):
        """Generate model comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = df_comparison['Model'].values
        colors = sns.color_palette("husl", len(models))
        
        # F1-Score comparison
        ax1 = axes[0, 0]
        ax1.barh(models, df_comparison['Test F1-Score'], color=colors, edgecolor='black')
        ax1.set_xlabel('F1-Score (%)', fontsize=11)
        ax1.set_title('Test F1-Score Comparison', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Accuracy comparison
        ax2 = axes[0, 1]
        ax2.barh(models, df_comparison['Test Accuracy'], color=colors, edgecolor='black')
        ax2.set_xlabel('Accuracy (%)', fontsize=11)
        ax2.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Training time comparison
        ax3 = axes[1, 0]
        ax3.barh(models, df_comparison['Training Time (s)'], color=colors, edgecolor='black')
        ax3.set_xlabel('Training Time (seconds)', fontsize=11)
        ax3.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Inference time comparison
        ax4 = axes[1, 1]
        ax4.barh(models, df_comparison['Inference (ms)'], color=colors, edgecolor='black')
        ax4.set_xlabel('Inference Time (ms/sample)', fontsize=11)
        ax4.set_title('Inference Speed Comparison', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/model_comparison.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved")
        plt.close()
    
    def save_best_model(self, criterion='f1_score'):
        """
        Save the best performing model
        
        Args:
            criterion: Metric to use for selection ('f1_score', 'accuracy', etc.)
        
        Returns:
            Name of best model
        """
        print("\n" + "="*80)
        print("SAVING BEST MODEL")
        print("="*80)
        
        if not self.trained_models:
            print("No models trained yet!")
            return None
        
        # Find best model
        best_score = -1
        best_model_name = None
        
        for model_name, metrics in self.performance_metrics.items():
            score = metrics['test'][criterion]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        print(f"\nBest model: {best_model_name}")
        print(f"Test {criterion}: {best_score*100:.2f}%")
        
        # Get best model
        best_model = self.trained_models[best_model_name]
        
        # Create version number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = "v1"
        
        # Save model
        model_filename = f"tii_ssrc23_{best_model_name.lower()}_{version}.pkl"
        model_path = f"{self.models_dir}/{model_filename}"
        
        joblib.dump(best_model, model_path)
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_name': best_model_name,
            'version': version,
            'timestamp': timestamp,
            'performance': self.performance_metrics[best_model_name],
            'training_time': self.training_times[best_model_name],
            'inference_time_per_sample': self.inference_times[best_model_name],
            'feature_count': self.X_train.shape[1],
            'class_count': len(self.label_to_attack),
            'training_samples': len(self.X_train),
            'attack_mapping': self.attack_mapping
        }
        
        metadata_path = f"{self.models_dir}/{best_model_name.lower()}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {metadata_path}")
        
        # Create a "best_model" symlink/copy
        best_model_path = f"{self.models_dir}/tii_ssrc23_best_model_v1.pkl"
        joblib.dump(best_model, best_model_path)
        print(f"✓ Best model copy saved to: {best_model_path}")
        
        return best_model_name
    
    def generate_training_summary(self):
        """Generate comprehensive training summary report"""
        print("\n" + "="*80)
        print("GENERATING TRAINING SUMMARY")
        print("="*80)
        
        summary = {
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_info': {
                'training_samples': int(len(self.X_train)),
                'validation_samples': int(len(self.X_val)),
                'test_samples': int(len(self.X_test)),
                'total_features': int(self.X_train.shape[1]),
                'attack_classes': len(self.label_to_attack),
                'class_names': list(self.attack_mapping.keys())
            },
            'models_trained': list(self.trained_models.keys()),
            'performance_metrics': {},
            'training_times': {},
            'best_model': None
        }
        
        # Add performance metrics for each model
        for model_name in self.trained_models.keys():
            summary['performance_metrics'][model_name] = {
                'test_accuracy': float(self.performance_metrics[model_name]['test']['accuracy']),
                'test_f1_score': float(self.performance_metrics[model_name]['test']['f1_score']),
                'test_precision': float(self.performance_metrics[model_name]['test']['precision']),
                'test_recall': float(self.performance_metrics[model_name]['test']['recall']),
                'inference_time_ms': float(self.inference_times[model_name] * 1000)
            }
            summary['training_times'][model_name] = float(self.training_times[model_name])
        
        # Identify best model
        best_f1 = -1
        for model_name, metrics in summary['performance_metrics'].items():
            if metrics['test_f1_score'] > best_f1:
                best_f1 = metrics['test_f1_score']
                summary['best_model'] = model_name
        
        # Save summary
        summary_path = f"{self.results_dir}/training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Training summary saved to: {summary_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"\nDataset: {summary['dataset_info']['training_samples']:,} train, "
              f"{summary['dataset_info']['test_samples']:,} test samples")
        print(f"Features: {summary['dataset_info']['total_features']}")
        print(f"Classes: {summary['dataset_info']['attack_classes']}")
        print(f"\nModels Trained: {len(summary['models_trained'])}")
        
        print("\nPerformance Summary:")
        for model_name, perf in summary['performance_metrics'].items():
            print(f"\n  {model_name}:")
            print(f"    Accuracy:  {perf['test_accuracy']*100:.2f}%")
            print(f"    F1-Score:  {perf['test_f1_score']*100:.2f}%")
            print(f"    Precision: {perf['test_precision']*100:.2f}%")
            print(f"    Recall:    {perf['test_recall']*100:.2f}%")
            print(f"    Inference: {perf['inference_time_ms']:.3f} ms/sample")
        
        print(f"\n🏆 Best Model: {summary['best_model']}")
        print(f"   F1-Score: {summary['performance_metrics'][summary['best_model']]['test_f1_score']*100:.2f}%")
        
        return summary


# Hyperparameter tuning utilities
class HyperparameterTuner:
    """
    Advanced hyperparameter tuning using Grid Search or Random Search
    """
    
    def __init__(self, trainer):
        """
        Initialize tuner with trainer instance
        
        Args:
            trainer: ModelTrainer instance
        """
        self.trainer = trainer
        self.best_params = {}
    
    def tune_xgboost(self, param_grid=None, n_iter=20):
        """
        Tune XGBoost hyperparameters using RandomizedSearchCV
        
        Args:
            param_grid: Dictionary of parameters to tune
            n_iter: Number of iterations for random search
        
        Returns:
            Best parameters
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING: XGBOOST")
        print("="*80)
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 150, 200, 250],
                'max_depth': [6, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
        
        print(f"\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Initialize model
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42,
            tree_method='hist'
        )
        
        # Perform random search
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        
        print(f"\nPerforming random search with {n_iter} iterations...")
        random_search.fit(self.trainer.X_train, self.trainer.y_train)
        
        print(f"\n✓ Tuning complete!")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        self.best_params['XGBoost'] = random_search.best_params_
        
        return random_search.best_params_
    
    def tune_lightgbm(self, param_grid=None, n_iter=20):
        """
        Tune LightGBM hyperparameters
        
        Args:
            param_grid: Dictionary of parameters to tune
            n_iter: Number of iterations
        
        Returns:
            Best parameters
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING: LIGHTGBM")
        print("="*80)
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 150, 200, 250],
                'max_depth': [10, 15, 20, 25],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70, 100],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_samples': [10, 20, 30]
            }
        
        print(f"\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Initialize model
        base_model = lgb.LGBMClassifier(
            objective='multiclass',
            metric='multi_logloss',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        
        # Perform random search
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        
        print(f"\nPerforming random search with {n_iter} iterations...")
        random_search.fit(self.trainer.X_train, self.trainer.y_train)
        
        print(f"\n✓ Tuning complete!")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        self.best_params['LightGBM'] = random_search.best_params_
        
        return random_search.best_params_
    
    def tune_random_forest(self, param_grid=None, n_iter=15):
        """
        Tune Random Forest hyperparameters
        
        Args:
            param_grid: Dictionary of parameters to tune
            n_iter: Number of iterations
        
        Returns:
            Best parameters
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING: RANDOM FOREST")
        print("="*80)
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 150, 200, 250, 300],
                'max_depth': [20, 30, 40, 50, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        
        print(f"\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Initialize model
        base_model = RandomForestClassifier(
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        # Perform random search
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        
        print(f"\nPerforming random search with {n_iter} iterations...")
        random_search.fit(self.trainer.X_train, self.trainer.y_train)
        
        print(f"\n✓ Tuning complete!")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        self.best_params['RandomForest'] = random_search.best_params_
        
        return random_search.best_params_


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("SENTINET - PHASE 2: MODEL TRAINING & OPTIMIZATION")
    print("="*80)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(
            data_dir='processed_data',
            models_dir='models',
            results_dir='results'
        )
        
        # Option 1: Train all models with default parameters
        print("\n" + "="*80)
        print("OPTION 1: TRAINING ALL MODELS (DEFAULT PARAMETERS)")
        print("="*80)
        
        trained_models = trainer.train_all_models(perform_cv=True)
        
        # Compare models
        comparison = trainer.compare_models()
        
        # Save best model
        best_model = trainer.save_best_model(criterion='f1_score')
        
        # Generate summary
        summary = trainer.generate_training_summary()
        
        print("\n" + "="*80)
        print("PHASE 2 COMPLETE ✓")
        print("="*80)
        print("\nDeliverables:")
        print(f"  • Trained models: {len(trained_models)}")
        print(f"  • Best model saved: models/tii_ssrc23_best_model_v1.pkl")
        print(f"  • Performance metrics: results/model_comparison.csv")
        print(f"  • Confusion matrices: results/plots/")
        print(f"  • Training summary: results/training_summary.json")
        
        print("\n" + "="*80)
        print("OPTIONAL: HYPERPARAMETER TUNING")
        print("="*80)
        print("\nTo perform hyperparameter tuning (recommended for production):")
        print("  1. Uncomment the tuning section below")
        print("  2. Run the script again")
        print("  3. This will take longer but may improve performance")
        
        # Optional: Hyperparameter tuning (uncomment to use)
        """
        print("\n" + "="*80)
        print("PERFORMING HYPERPARAMETER TUNING")
        print("="*80)
        
        tuner = HyperparameterTuner(trainer)
        
        # Tune XGBoost
        xgb_params = tuner.tune_xgboost(n_iter=20)
        
        # Tune LightGBM
        lgb_params = tuner.tune_lightgbm(n_iter=20)
        
        # Retrain with best parameters
        print("\n" + "="*80)
        print("RETRAINING WITH OPTIMIZED PARAMETERS")
        print("="*80)
        
        trainer.train_xgboost(**xgb_params)
        trainer.train_lightgbm(**lgb_params)
        
        # Compare again
        trainer.compare_models()
        trainer.save_best_model(criterion='f1_score')
        """
        
        print("\n✅ Model training complete!")
        print("\nNext steps:")
        print("  1. Review model comparison in results/model_comparison.csv")
        print("  2. Analyze confusion matrices in results/plots/")
        print("  3. Proceed to Phase 3: Real-Time Inference Module")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠ Please run preprocessing.py (Phase 1) first!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()