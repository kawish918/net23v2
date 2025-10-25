"""
SentiNet - Deep Learning Model Training
Implements neural network architectures for intrusion detection

This module provides:
- Multi-layer Perceptron (MLP)
- 1D Convolutional Neural Network (CNN)
- LSTM-based models
- Attention mechanisms

Using TensorFlow/Keras for implementation
"""

import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow not available. Install with: pip install tensorflow")


class DeepLearningTrainer:
    """
    Deep learning model trainer for intrusion detection
    """
    
    def __init__(self, data_dir='processed_data', models_dir='models', results_dir='results'):
        """
        Initialize deep learning trainer
        
        Args:
            data_dir: Directory with preprocessed data
            models_dir: Directory for saving models
            results_dir: Directory for results
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for deep learning models")
        
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        
        # Load data
        self._load_data()
        self._load_mappings()
        
        # Prepare data for deep learning
        self._prepare_dl_data()
        
        self.trained_models = {}
        self.history = {}
        
        print("="*80)
        print("DEEP LEARNING TRAINER INITIALIZED")
        print("="*80)
        print(f"  Training samples: {len(self.X_train):,}")
        print(f"  Features: {self.X_train.shape[1]}")
        print(f"  Classes: {self.num_classes}")
    
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
    
    def _prepare_dl_data(self):
        """Prepare data for deep learning (one-hot encoding, reshaping)"""
        # Get number of classes
        self.num_classes = len(np.unique(self.y_train))
        
        # One-hot encode labels
        self.y_train_cat = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_val_cat = to_categorical(self.y_val, num_classes=self.num_classes)
        self.y_test_cat = to_categorical(self.y_test, num_classes=self.num_classes)
        
        # Store input shape
        self.input_shape = self.X_train.shape[1]
        
        print(f"\n✓ Data prepared for deep learning")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Output classes: {self.num_classes}")
    
    def build_mlp_model(self, hidden_layers=[256, 128, 64], dropout_rate=0.3):
        """
        Build Multi-Layer Perceptron model
        
        Args:
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*80)
        print("BUILDING MLP MODEL")
        print("="*80)
        
        model = models.Sequential(name='MLP')
        
        # Input layer
        model.add(layers.Input(shape=(self.input_shape,)))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(average='weighted')]
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def build_cnn_model(self, filters=[64, 128, 256], kernel_size=3, dropout_rate=0.3):
        """
        Build 1D Convolutional Neural Network
        
        Args:
            filters: List of filter counts for conv layers
            kernel_size: Size of convolutional kernels
            dropout_rate: Dropout rate
        
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*80)
        print("BUILDING 1D CNN MODEL")
        print("="*80)
        
        model = models.Sequential(name='CNN_1D')
        
        # Reshape input for 1D conv
        model.add(layers.Input(shape=(self.input_shape, 1)))
        
        # Convolutional blocks
        for i, n_filters in enumerate(filters):
            model.add(layers.Conv1D(
                n_filters, kernel_size, 
                activation='relu', 
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_conv_{i+1}'))
            model.add(layers.MaxPooling1D(2, name=f'maxpool_{i+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}'))
        
        # Flatten and dense layers
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(128, activation='relu', name='dense_1'))
        model.add(layers.BatchNormalization(name='bn_dense'))
        model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(average='weighted')]
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def build_lstm_model(self, lstm_units=[128, 64], dropout_rate=0.3):
        """
        Build LSTM-based model
        
        Args:
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate
        
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*80)
        print("BUILDING LSTM MODEL")
        print("="*80)
        
        model = models.Sequential(name='LSTM')
        
        # Reshape for LSTM (samples, timesteps, features)
        model.add(layers.Input(shape=(self.input_shape, 1)))
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                name=f'lstm_{i+1}'
            ))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_lstm_{i+1}'))
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(average='weighted')]
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def build_attention_model(self, units=128, num_heads=4, dropout_rate=0.3):
        """
        Build model with multi-head attention mechanism
        
        Args:
            units: Number of units in attention layer
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*80)
        print("BUILDING ATTENTION MODEL")
        print("="*80)
        
        inputs = layers.Input(shape=(self.input_shape,))
        
        # Reshape for attention
        x = layers.Reshape((self.input_shape, 1))(inputs)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads,
            name='multi_head_attention'
        )(x, x)
        
        x = layers.Add()([x, attention_output])  # Residual connection
        x = layers.LayerNormalization(name='layer_norm_1')(x)
        
        # Feed-forward network
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_2')(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='Attention')
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(average='weighted')]
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train_model(self, model, model_name, epochs=50, batch_size=128, 
                   patience=10, use_class_weights=True):
        """
        Train deep learning model
        
        Args:
            model: Compiled Keras model
            model_name: Name identifier
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            use_class_weights: Whether to use class weights for imbalance
        
        Returns:
            Training history
        """
        print(f"\n{'='*80}")
        print(f"TRAINING: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Prepare data based on model type
        if 'CNN' in model_name or 'LSTM' in model_name:
            X_train = self.X_train.reshape(-1, self.input_shape, 1)
            X_val = self.X_val.reshape(-1, self.input_shape, 1)
            X_test = self.X_test.reshape(-1, self.input_shape, 1)
        else:
            X_train = self.X_train
            X_val = self.X_val
            X_test = self.X_test
        
        # Calculate class weights if needed
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights_array = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            class_weights = dict(enumerate(class_weights_array))
            print(f"\nUsing class weights for imbalanced data")
        else:
            class_weights = None
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f"{self.models_dir}/temp_{model_name}_best.keras",
                monitor='val_f1_score',
                mode='max',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train
        print(f"\nTraining with:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Early stopping patience: {patience}")
        
        start_time = time.time()
        
        history = model.fit(
            X_train, self.y_train_cat,
            validation_data=(X_val, self.y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            class_weight=class_weights,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        print(f"\nEvaluating on test set...")
        test_loss, test_acc, test_f1 = model.evaluate(X_test, self.y_test_cat, verbose=0)
        
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
        print(f"  Test F1-Score: {test_f1*100:.2f}%")
        
        # Store model and history
        self.trained_models[model_name] = {
            'model': model,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'training_time': training_time
        }
        self.history[model_name] = history.history
        
        # Plot training history
        self._plot_training_history(history, model_name)
        
        # Generate confusion matrix
        self._generate_confusion_matrix(model, model_name, X_test)
        
        return history
    
    def _plot_training_history(self, history, model_name):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1 = axes[0]
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title(f'{model_name} - Training Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2 = axes[1]
        ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title(f'{model_name} - Training Accuracy', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/{model_name}_training_history.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved")
        plt.close()
    
    def _generate_confusion_matrix(self, model, model_name, X_test):
        """Generate confusion matrix for deep learning model"""
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        labels = [self.label_to_attack[i] for i in range(len(cm))]
        
        # Raw counts
        ax1 = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        ax1.set_title(f'{model_name} - Confusion Matrix', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=11)
        ax1.set_xlabel('Predicted Label', fontsize=11)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)
        
        # Normalized
        ax2 = axes[1]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Percentage'})
        ax2.set_title(f'{model_name} - Normalized Confusion Matrix', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=11)
        ax2.set_xlabel('Predicted Label', fontsize=11)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/{model_name}_confusion_matrix.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved")
        plt.close()
    
    def train_all_models(self, epochs=50, batch_size=128):
        """Train all deep learning models"""
        print("\n" + "="*80)
        print("TRAINING ALL DEEP LEARNING MODELS")
        print("="*80)
        
        models_to_train = [
            ('MLP', lambda: self.build_mlp_model()),
            ('CNN_1D', lambda: self.build_cnn_model()),
            ('LSTM', lambda: self.build_lstm_model()),
            ('Attention', lambda: self.build_attention_model())
        ]
        
        for model_name, build_func in models_to_train:
            try:
                model = build_func()
                self.train_model(model, model_name, epochs=epochs, batch_size=batch_size)
            except Exception as e:
                print(f"\n✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.trained_models
    
    def compare_models(self):
        """Compare all trained deep learning models"""
        print("\n" + "="*80)
        print("DEEP LEARNING MODEL COMPARISON")
        print("="*80)
        
        if not self.trained_models:
            print("No models trained yet!")
            return
        
        import pandas as pd
        
        comparison_data = []
        for model_name, model_info in self.trained_models.items():
            comparison_data.append({
                'Model': model_name,
                'Test Accuracy': model_info['test_acc'] * 100,
                'Test F1-Score': model_info['test_f1'] * 100,
                'Training Time (s)': model_info['training_time'],
                'Parameters': model_info['model'].count_params()
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Test F1-Score', ascending=False)
        
        print("\n" + df_comparison.to_string(index=False))
        
        # Save comparison
        df_comparison.to_csv(f"{self.results_dir}/dl_model_comparison.csv", index=False)
        print(f"\n✓ Comparison saved")
        
        # Plot comparison
        self._plot_model_comparison(df_comparison)
        
        best_model = df_comparison.iloc[0]['Model']
        best_f1 = df_comparison.iloc[0]['Test F1-Score']
        
        print(f"\n🏆 BEST DL MODEL: {best_model}")
        print(f"   Test F1-Score: {best_f1:.2f}%")
        
        return df_comparison
    
    def _plot_model_comparison(self, df_comparison):
        """Plot deep learning model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = df_comparison['Model'].values
        colors = sns.color_palette("Set2", len(models))
        
        # F1-Score
        ax1 = axes[0, 0]
        ax1.barh(models, df_comparison['Test F1-Score'], color=colors, edgecolor='black')
        ax1.set_xlabel('F1-Score (%)', fontsize=11)
        ax1.set_title('Test F1-Score Comparison', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Accuracy
        ax2 = axes[0, 1]
        ax2.barh(models, df_comparison['Test Accuracy'], color=colors, edgecolor='black')
        ax2.set_xlabel('Accuracy (%)', fontsize=11)
        ax2.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Training time
        ax3 = axes[1, 0]
        ax3.barh(models, df_comparison['Training Time (s)'], color=colors, edgecolor='black')
        ax3.set_xlabel('Training Time (seconds)', fontsize=11)
        ax3.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Parameters
        ax4 = axes[1, 1]
        ax4.barh(models, df_comparison['Parameters'] / 1000, color=colors, edgecolor='black')
        ax4.set_xlabel('Parameters (K)', fontsize=11)
        ax4.set_title('Model Complexity (Parameters)', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/dl_model_comparison.png",
                   dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved")
        plt.close()
    
    def save_best_model(self):
        """Save the best performing deep learning model"""
        print("\n" + "="*80)
        print("SAVING BEST DEEP LEARNING MODEL")
        print("="*80)
        
        if not self.trained_models:
            print("No models trained yet!")
            return None
        
        # Find best model by F1-score
        best_f1 = -1
        best_name = None
        
        for name, info in self.trained_models.items():
            if info['test_f1'] > best_f1:
                best_f1 = info['test_f1']
                best_name = name
        
        print(f"\nBest model: {best_name}")
        print(f"Test F1-Score: {best_f1*100:.2f}%")
        
        best_model = self.trained_models[best_name]['model']
        
        # Save model
        model_path = f"{self.models_dir}/tii_ssrc23_dl_{best_name.lower()}_v1.keras"
        best_model.save(model_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': best_name,
            'architecture': 'deep_learning',
            'test_accuracy': float(self.trained_models[best_name]['test_acc']),
            'test_f1_score': float(self.trained_models[best_name]['test_f1']),
            'training_time': float(self.trained_models[best_name]['training_time']),
            'parameters': int(best_model.count_params()),
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'attack_mapping': self.attack_mapping
        }
        
        metadata_path = f"{self.models_dir}/dl_{best_name.lower()}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return best_name


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("SENTINET - DEEP LEARNING MODEL TRAINING")
    print("="*80)
    
    if not TF_AVAILABLE:
        print("\n❌ TensorFlow not available!")
        print("Install with: pip install tensorflow")
        exit(1)
    
    try:
        # Initialize trainer
        trainer = DeepLearningTrainer(
            data_dir='processed_data',
            models_dir='models',
            results_dir='results'
        )
        
        # Train all models
        print("\n" + "="*80)
        print("TRAINING ALL DEEP LEARNING MODELS")
        print("="*80)
        print("\nNote: This may take 10-30 minutes depending on hardware")
        
        trained_models = trainer.train_all_models(epochs=50, batch_size=128)
        
        # Compare models
        comparison = trainer.compare_models()
        
        # Save best model
        best_model = trainer.save_best_model()
        
        print("\n" + "="*80)
        print("DEEP LEARNING TRAINING COMPLETE ✓")
        print("="*80)
        print(f"\nTrained {len(trained_models)} models:")
        for name in trained_models.keys():
            print(f"  • {name}")
        
        print(f"\nBest model: {best_model}")
        print(f"\nDeliverables:")
        print(f"  • DL models saved in models/")
        print(f"  • Training histories: results/plots/*_training_history.png")
        print(f"  • Confusion matrices: results/plots/*_confusion_matrix.png")
        print(f"  • Comparison: results/dl_model_comparison.csv")
        
        print("\n✅ Deep learning training complete!")
        print("\nDeep learning models offer:")
        print("  • Automatic feature learning")
        print("  • Better handling of complex patterns")
        print("  • State-of-the-art performance on large datasets")
        print("  • Attention mechanisms for interpretability")
        
        print("\n💡 Tips:")
        print("  • MLP: Fast, good baseline")
        print("  • CNN: Excellent for pattern detection")
        print("  • LSTM: Good for sequential patterns")
        print("  • Attention: Best interpretability + performance")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠ Please run preprocessing.py (Phase 1) first!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()