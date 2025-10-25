"""
SentiNet - Advanced Ensemble Model Training
Implements stacking, voting, and weighted ensemble methods

This module creates high-performance ensemble models by combining
predictions from multiple base learners for superior accuracy.
"""

import numpy as np
import joblib
import json
import os
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class EnsembleModelTrainer:
    """
    Advanced ensemble model training and evaluation
    """
    
    def __init__(self, data_dir='processed_data', models_dir='models', results_dir='results'):
        """
        Initialize ensemble trainer
        
        Args:
            data_dir: Directory with preprocessed data
            models_dir: Directory for saving models
            results_dir: Directory for results
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Load data
        self._load_data()
        self._load_mappings()
        
        self.ensemble_models = {}
        
        print("="*80)
        print("ENSEMBLE MODEL TRAINER INITIALIZED")
        print("="*80)
    
    def _load_data(self):
        """Load preprocessed data"""
        data = np.load(f"{self.data_dir}/dataset_splits.npz")
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
    
    def _load_mappings(self):
        """Load label mappings"""
        with open(f"{self.data_dir}/attack_mapping.json", 'r') as f:
            self.attack_mapping = json.load(f)
        self.label_to_attack = {v: k for k, v in self.attack_mapping.items()}
    
    def create_voting_ensemble(self, voting='soft'):
        """
        Create voting ensemble combining multiple models
        
        Args:
            voting: 'hard' or 'soft' voting
        
        Returns:
            Trained voting classifier
        """
        print("\n" + "="*80)
        print(f"TRAINING VOTING ENSEMBLE ({voting.upper()} VOTING)")
        print("="*80)
        
        # Define base estimators with optimized parameters
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                n_jobs=-1,
                random_state=42,
                tree_method='hist'
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced',
                verbose=-1
            ))
        ]
        
        print(f"\nBase estimators: {len(estimators)}")
        for name, _ in estimators:
            print(f"  - {name}")
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        # Train
        print(f"\nTraining {voting} voting ensemble...")
        voting_clf.fit(self.X_train, self.y_train)
        
        # Evaluate
        print("\nEvaluating on validation set...")
        y_pred_val = voting_clf.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, y_pred_val)
        val_f1 = f1_score(self.y_val, y_pred_val, average='weighted')
        
        print(f"  Validation Accuracy: {val_acc*100:.2f}%")
        print(f"  Validation F1-Score: {val_f1*100:.2f}%")
        
        print("\nEvaluating on test set...")
        y_pred_test = voting_clf.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
        
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
        print(f"  Test F1-Score: {test_f1*100:.2f}%")
        
        # Store model
        model_name = f'voting_{voting}'
        self.ensemble_models[model_name] = {
            'model': voting_clf,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_f1': test_f1
        }
        
        print(f"\n✓ Voting ensemble trained successfully")
        
        return voting_clf
    
    def create_stacking_ensemble(self, meta_learner='logistic'):
        """
        Create stacking ensemble with meta-learner
        
        Args:
            meta_learner: Type of meta-learner ('logistic', 'xgboost', 'lightgbm')
        
        Returns:
            Trained stacking classifier
        """
        print("\n" + "="*80)
        print(f"TRAINING STACKING ENSEMBLE (Meta-learner: {meta_learner.upper()})")
        print("="*80)
        
        # Define base estimators
        base_estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                n_jobs=-1,
                random_state=42,
                tree_method='hist'
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced',
                verbose=-1
            ))
        ]
        
        # Define meta-learner
        if meta_learner == 'logistic':
            final_estimator = LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                n_jobs=-1,
                random_state=42
            )
        elif meta_learner == 'xgboost':
            final_estimator = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='multi:softprob',
                n_jobs=-1,
                random_state=42
            )
        elif meta_learner == 'lightgbm':
            final_estimator = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner}")
        
        print(f"\nBase estimators: {len(base_estimators)}")
        for name, _ in base_estimators:
            print(f"  - {name}")
        print(f"Meta-learner: {meta_learner}")
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=3,
            n_jobs=-1
        )
        
        # Train
        print(f"\nTraining stacking ensemble...")
        stacking_clf.fit(self.X_train, self.y_train)
        
        # Evaluate
        print("\nEvaluating on validation set...")
        y_pred_val = stacking_clf.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, y_pred_val)
        val_f1 = f1_score(self.y_val, y_pred_val, average='weighted')
        
        print(f"  Validation Accuracy: {val_acc*100:.2f}%")
        print(f"  Validation F1-Score: {val_f1*100:.2f}%")
        
        print("\nEvaluating on test set...")
        y_pred_test = stacking_clf.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
        
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
        print(f"  Test F1-Score: {test_f1*100:.2f}%")
        
        # Store model
        model_name = f'stacking_{meta_learner}'
        self.ensemble_models[model_name] = {
            'model': stacking_clf,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_f1': test_f1
        }
        
        print(f"\n✓ Stacking ensemble trained successfully")
        
        return stacking_clf
    
    def create_weighted_ensemble(self, weights=None):
        """
        Create weighted ensemble with custom weights
        
        Args:
            weights: List of weights for each model [rf_weight, xgb_weight, lgb_weight]
                    If None, uses equal weights
        
        Returns:
            Custom weighted ensemble predictor
        """
        print("\n" + "="*80)
        print("TRAINING WEIGHTED ENSEMBLE")
        print("="*80)
        
        # Train base models
        print("\nTraining base models...")
        
        # Random Forest
        print("  Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(self.X_train, self.y_train)
        
        # XGBoost
        print("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            n_jobs=-1,
            random_state=42,
            tree_method='hist'
        )
        xgb_model.fit(self.X_train, self.y_train)
        
        # LightGBM
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=15,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        lgb_model.fit(self.X_train, self.y_train)
        
        # Determine optimal weights if not provided
        if weights is None:
            print("\nDetermining optimal weights using validation set...")
            
            # Get individual model accuracies
            rf_f1 = f1_score(self.y_val, rf_model.predict(self.X_val), average='weighted')
            xgb_f1 = f1_score(self.y_val, xgb_model.predict(self.X_val), average='weighted')
            lgb_f1 = f1_score(self.y_val, lgb_model.predict(self.X_val), average='weighted')
            
            # Weight by performance (normalize to sum to 1)
            total_f1 = rf_f1 + xgb_f1 + lgb_f1
            weights = [rf_f1/total_f1, xgb_f1/total_f1, lgb_f1/total_f1]
            
            print(f"  Optimal weights determined:")
            print(f"    RandomForest: {weights[0]:.3f} (F1: {rf_f1:.4f})")
            print(f"    XGBoost:      {weights[1]:.3f} (F1: {xgb_f1:.4f})")
            print(f"    LightGBM:     {weights[2]:.3f} (F1: {lgb_f1:.4f})")
        else:
            print(f"\nUsing provided weights: {weights}")
        
        # Create weighted ensemble predictor
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = np.array(weights)
            
            def predict_proba(self, X):
                # Get predictions from all models
                probas = []
                for model in self.models:
                    probas.append(model.predict_proba(X))
                
                # Weighted average
                weighted_proba = np.zeros_like(probas[0])
                for i, proba in enumerate(probas):
                    weighted_proba += self.weights[i] * proba
                
                return weighted_proba
            
            def predict(self, X):
                proba = self.predict_proba(X)
                return np.argmax(proba, axis=1)
        
        weighted_ensemble = WeightedEnsemble(
            models=[rf_model, xgb_model, lgb_model],
            weights=weights
        )
        
        # Evaluate
        print("\nEvaluating weighted ensemble...")
        print("\nValidation set:")
        y_pred_val = weighted_ensemble.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, y_pred_val)
        val_f1 = f1_score(self.y_val, y_pred_val, average='weighted')
        print(f"  Accuracy: {val_acc*100:.2f}%")
        print(f"  F1-Score: {val_f1*100:.2f}%")
        
        print("\nTest set:")
        y_pred_test = weighted_ensemble.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
        print(f"  Accuracy: {test_acc*100:.2f}%")
        print(f"  F1-Score: {test_f1*100:.2f}%")
        
        # Store model
        self.ensemble_models['weighted'] = {
            'model': weighted_ensemble,
            'weights': weights,
            'base_models': {'rf': rf_model, 'xgb': xgb_model, 'lgb': lgb_model},
            'val_acc': val_acc,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_f1': test_f1
        }
        
        print(f"\n✓ Weighted ensemble trained successfully")
        
        return weighted_ensemble
    
    def train_all_ensembles(self):
        """Train all ensemble methods and compare"""
        print("\n" + "="*80)
        print("TRAINING ALL ENSEMBLE METHODS")
        print("="*80)
        
        # Voting ensembles
        self.create_voting_ensemble(voting='soft')
        self.create_voting_ensemble(voting='hard')
        
        # Stacking ensembles
        self.create_stacking_ensemble(meta_learner='logistic')
        self.create_stacking_ensemble(meta_learner='xgboost')
        
        # Weighted ensemble
        self.create_weighted_ensemble()
        
        return self.ensemble_models
    
    def compare_ensembles(self):
        """Compare all trained ensemble models"""
        print("\n" + "="*80)
        print("ENSEMBLE MODEL COMPARISON")
        print("="*80)
        
        if not self.ensemble_models:
            print("No ensemble models trained yet!")
            return
        
        # Create comparison table
        comparison_data = []
        
        for model_name, model_info in self.ensemble_models.items():
            comparison_data.append({
                'Ensemble Type': model_name,
                'Val Accuracy': model_info['val_acc'] * 100,
                'Val F1-Score': model_info['val_f1'] * 100,
                'Test Accuracy': model_info['test_acc'] * 100,
                'Test F1-Score': model_info['test_f1'] * 100
            })
        
        import pandas as pd
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Test F1-Score', ascending=False)
        
        print("\n" + df_comparison.to_string(index=False))
        
        # Save comparison
        df_comparison.to_csv(f"{self.results_dir}/ensemble_comparison.csv", index=False)
        print(f"\n✓ Comparison saved to {self.results_dir}/ensemble_comparison.csv")
        
        # Visualize comparison
        self._plot_ensemble_comparison(df_comparison)
        
        # Identify best ensemble
        best_ensemble = df_comparison.iloc[0]['Ensemble Type']
        best_f1 = df_comparison.iloc[0]['Test F1-Score']
        
        print(f"\n🏆 BEST ENSEMBLE: {best_ensemble}")
        print(f"   Test F1-Score: {best_f1:.2f}%")
        
        return df_comparison
    
    def _plot_ensemble_comparison(self, df_comparison):
        """Plot ensemble model comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ensembles = df_comparison['Ensemble Type'].values
        colors = sns.color_palette("Set2", len(ensembles))
        
        # F1-Score comparison
        ax1 = axes[0]
        x = np.arange(len(ensembles))
        width = 0.35
        
        ax1.bar(x - width/2, df_comparison['Val F1-Score'], width, 
               label='Validation', color='skyblue', edgecolor='black')
        ax1.bar(x + width/2, df_comparison['Test F1-Score'], width,
               label='Test', color='lightcoral', edgecolor='black')
        
        ax1.set_xlabel('Ensemble Type', fontsize=11)
        ax1.set_ylabel('F1-Score (%)', fontsize=11)
        ax1.set_title('Ensemble F1-Score Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(ensembles, rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Accuracy comparison
        ax2 = axes[1]
        ax2.bar(x - width/2, df_comparison['Val Accuracy'], width,
               label='Validation', color='lightgreen', edgecolor='black')
        ax2.bar(x + width/2, df_comparison['Test Accuracy'], width,
               label='Test', color='plum', edgecolor='black')
        
        ax2.set_xlabel('Ensemble Type', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.set_title('Ensemble Accuracy Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(ensembles, rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/ensemble_comparison.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Ensemble comparison plot saved")
        plt.close()
    
    def save_best_ensemble(self):
        """Save the best performing ensemble model"""
        print("\n" + "="*80)
        print("SAVING BEST ENSEMBLE MODEL")
        print("="*80)
        
        if not self.ensemble_models:
            print("No ensemble models trained yet!")
            return None
        
        # Find best ensemble by test F1-score
        best_f1 = -1
        best_name = None
        
        for name, info in self.ensemble_models.items():
            if info['test_f1'] > best_f1:
                best_f1 = info['test_f1']
                best_name = name
        
        print(f"\nBest ensemble: {best_name}")
        print(f"Test F1-Score: {best_f1*100:.2f}%")
        
        best_ensemble = self.ensemble_models[best_name]
        
        # Save model
        model_path = f"{self.models_dir}/tii_ssrc23_ensemble_{best_name}_v1.pkl"
        
        if best_name == 'weighted':
            # Save weighted ensemble with base models
            save_data = {
                'ensemble': best_ensemble['model'],
                'base_models': best_ensemble['base_models'],
                'weights': best_ensemble['weights']
            }
            joblib.dump(save_data, model_path)
        else:
            joblib.dump(best_ensemble['model'], model_path)
        
        print(f"\n✓ Ensemble saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'ensemble_type': best_name,
            'test_accuracy': best_ensemble['test_acc'],
            'test_f1_score': best_ensemble['test_f1'],
            'val_accuracy': best_ensemble['val_acc'],
            'val_f1_score': best_ensemble['val_f1'],
            'attack_mapping': self.attack_mapping
        }
        
        metadata_path = f"{self.models_dir}/ensemble_{best_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return best_name
    
    def generate_detailed_report(self, ensemble_name):
        """Generate detailed classification report for specific ensemble"""
        if ensemble_name not in self.ensemble_models:
            print(f"Ensemble {ensemble_name} not found!")
            return
        
        print(f"\n{'='*80}")
        print(f"DETAILED REPORT: {ensemble_name.upper()}")
        print(f"{'='*80}")
        
        model = self.ensemble_models[ensemble_name]['model']
        
        # Get predictions
        y_pred = model.predict(self.X_test)
        
        # Classification report
        report = classification_report(
            self.y_test, y_pred,
            target_names=[self.label_to_attack[i] for i in sorted(self.label_to_attack.keys())],
            zero_division=0
        )
        
        print("\nClassification Report:")
        print(report)
        
        # Save report
        report_path = f"{self.results_dir}/ensemble_{ensemble_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: {report_path}")


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("SENTINET - ADVANCED ENSEMBLE MODEL TRAINING")
    print("="*80)
    
    try:
        # Initialize trainer
        trainer = EnsembleModelTrainer(
            data_dir='processed_data',
            models_dir='models',
            results_dir='results'
        )
        
        # Train all ensembles
        print("\n" + "="*80)
        print("TRAINING ALL ENSEMBLE METHODS")
        print("="*80)
        
        ensembles = trainer.train_all_ensembles()
        
        # Compare ensembles
        comparison = trainer.compare_ensembles()
        
        # Save best ensemble
        best_ensemble = trainer.save_best_ensemble()
        
        # Generate detailed report for best ensemble
        if best_ensemble:
            trainer.generate_detailed_report(best_ensemble)
        
        print("\n" + "="*80)
        print("ENSEMBLE TRAINING COMPLETE ✓")
        print("="*80)
        print(f"\nTrained {len(ensembles)} ensemble models:")
        for name in ensembles.keys():
            print(f"  • {name}")
        
        print(f"\nBest ensemble: {best_ensemble}")
        print(f"\nDeliverables:")
        print(f"  • Ensemble models saved in models/")
        print(f"  • Comparison report: results/ensemble_comparison.csv")
        print(f"  • Visualization: results/plots/ensemble_comparison.png")
        
        print("\n✅ Ensemble training complete!")
        print("\nEnsemble models typically provide:")
        print("  • Higher accuracy than individual models")
        print("  • Better generalization")
        print("  • More robust predictions")
        print("  • Reduced overfitting")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠ Please run preprocessing.py (Phase 1) first!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()