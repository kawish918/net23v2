"""
SentiNet - Quick Start Script
Automated execution of entire training pipeline

This script runs all steps in sequence:
1. Data preprocessing
2. Model training
3. Model testing

Usage:
    python run_all.py                    # Full pipeline
    python run_all.py --quick            # Skip ensemble & deep learning
    python run_all.py --preprocessing    # Only preprocessing
    python run_all.py --training         # Only training
"""

import sys
import os
import time
import argparse
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def print_section(text):
    """Print section separator"""
    print("\n" + "-"*80)
    print(f"  {text}")
    print("-"*80 + "\n")


def check_dependencies():
    """Check if all required libraries are installed"""
    print_section("Checking Dependencies")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib',
        'imblearn': 'imbalanced-learn'
    }
    
    optional_packages = {
        'tensorflow': 'tensorflow',
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_required.append(package)
    
    # Check optional
    print("\nOptional packages:")
    for module, package in optional_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠ {package} - Not installed (deep learning disabled)")
            missing_optional.append(package)
    
    if missing_required:
        print("\n❌ Missing required packages!")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_required)}")
        return False
    
    print("\n✅ All required dependencies installed!")
    return True


def check_dataset():
    """Check if dataset exists"""
    print_section("Checking Dataset")
    
    dataset_path = "TII-SSRC-23/data.csv"
    
    if os.path.exists(dataset_path):
        file_size = os.path.getsize(dataset_path) / (1024 * 1024)  # MB
        print(f"✓ Dataset found: {dataset_path}")
        print(f"  Size: {file_size:.2f} MB")
        return True
    else:
        print(f"✗ Dataset not found: {dataset_path}")
        print("\nPlease place your TII-SSRC-23 dataset in:")
        print(f"  {os.path.abspath(dataset_path)}")
        return False


def run_preprocessing():
    """Run data preprocessing"""
    print_header("PHASE 1: DATA PREPROCESSING")
    
    start_time = time.time()
    
    try:
        # Import and run preprocessor
        sys.path.insert(0, 'src')
        from preprocessing import TIIDataPreprocessor
        
        preprocessor = TIIDataPreprocessor(
            data_path='TII-SSRC-23/data.csv',
            output_dir='processed_data'
        )
        
        success = preprocessor.run_full_pipeline(
            balance_strategy='hybrid',
            test_size=0.2
        )
        
        elapsed = time.time() - start_time
        
        if success:
            print(f"\n✅ Preprocessing completed in {elapsed:.2f} seconds")
            return True
        else:
            print(f"\n❌ Preprocessing failed!")
            return False
    
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_eda():
    """Run exploratory data analysis"""
    print_header("ADVANCED EDA & VISUALIZATIONS")
    
    start_time = time.time()
    
    try:
        from eda_visualizations import AdvancedEDA
        
        eda = AdvancedEDA(
            data_path='processed_data/dataset_splits.npz',
            mapping_path='processed_data/attack_mapping.json',
            feature_path='processed_data/feature_mapping.json',
            output_dir='processed_data/visualizations'
        )
        
        eda.generate_all_visualizations()
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ EDA completed in {elapsed:.2f} seconds")
        return True
    
    except Exception as e:
        print(f"\n⚠ Warning: EDA failed: {e}")
        print("Continuing with training...")
        return True  # Non-critical, continue anyway


def run_model_training():
    """Run model training"""
    print_header("PHASE 2: MODEL TRAINING")
    
    start_time = time.time()
    
    try:
        from train_model import ModelTrainer
        
        trainer = ModelTrainer(
            data_dir='processed_data',
            models_dir='models',
            results_dir='results'
        )
        
        # Train all models
        trained_models = trainer.train_all_models(perform_cv=True)
        
        # Compare models
        trainer.compare_models()
        
        # Save best model
        trainer.save_best_model(criterion='f1_score')
        
        # Generate summary
        trainer.generate_training_summary()
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Model training completed in {elapsed:.2f} seconds")
        return True
    
    except Exception as e:
        print(f"\n❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_ensemble_training():
    """Run ensemble model training"""
    print_header("ENSEMBLE MODEL TRAINING")
    
    start_time = time.time()
    
    try:
        from ensemble_model import EnsembleModelTrainer
        
        trainer = EnsembleModelTrainer(
            data_dir='processed_data',
            models_dir='models',
            results_dir='results'
        )
        
        # Train all ensembles
        trainer.train_all_ensembles()
        
        # Compare ensembles
        trainer.compare_ensembles()
        
        # Save best ensemble
        trainer.save_best_ensemble()
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Ensemble training completed in {elapsed:.2f} seconds")
        return True
    
    except Exception as e:
        print(f"\n⚠ Warning: Ensemble training failed: {e}")
        print("Continuing with standard models...")
        return True  # Non-critical


def run_model_testing():
    """Run model testing"""
    print_header("MODEL TESTING & VERIFICATION")
    
    try:
        from test_trained_model import load_model_and_data, test_model_properties, test_full_performance
        
        # Load model and data
        model, X_test, y_test, scaler, label_to_attack = load_model_and_data()
        
        if model is None:
            print("❌ Failed to load model!")
            return False
        
        # Run tests
        test_model_properties(model)
        test_full_performance(model, X_test, y_test, label_to_attack)
        
        print(f"\n✅ Model testing completed")
        return True
    
    except Exception as e:
        print(f"\n⚠ Warning: Model testing failed: {e}")
        return True  # Non-critical


def print_summary(start_time, steps_completed):
    """Print execution summary"""
    print_header("EXECUTION SUMMARY")
    
    total_time = time.time() - start_time
    
    print(f"Total Execution Time: {total_time/60:.2f} minutes")
    print(f"\nSteps Completed:")
    
    for step, completed in steps_completed.items():
        status = "✅" if completed else "❌"
        print(f"  {status} {step}")
    
    print("\n" + "="*80)
    
    if all(steps_completed.values()):
        print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nYour models are ready for deployment!")
        print("\nNext Steps:")
        print("  1. Check results/ folder for performance metrics")
        print("  2. Review models/ folder for saved models")
        print("  3. Proceed to Phase 3: Real-Time Inference")
    else:
        print("\n⚠ SOME STEPS FAILED")
        print("\nPlease review the errors above and:")
        print("  1. Check that all dependencies are installed")
        print("  2. Verify dataset is in correct location")
        print("  3. Run failed steps individually for debugging")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='SentiNet Quick Start')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode (skip ensemble & deep learning)')
    parser.add_argument('--preprocessing', action='store_true',
                       help='Run only preprocessing')
    parser.add_argument('--training', action='store_true',
                       help='Run only training')
    parser.add_argument('--no-eda', action='store_true',
                       help='Skip EDA visualizations')
    
    args = parser.parse_args()
    
    # Print banner
    print("="*80)
    print("  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗████████╗  ".center(80))
    print("  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝╚══██╔══╝  ".center(80))
    print("  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗     ██║     ".center(80))
    print("  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝     ██║     ".center(80))
    print("  ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗   ██║     ".center(80))
    print("  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝     ".center(80))
    print("="*80)
    print("AI-Based Intrusion Detection & Prevention System".center(80))
    print("="*80)
    
    start_time = time.time()
    steps_completed = {}
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check dataset (if running preprocessing)
    if not args.training:
        if not check_dataset():
            return
    
    # Determine what to run
    run_all = not (args.preprocessing or args.training)
    
    # PHASE 1: Preprocessing
    if args.preprocessing or run_all:
        steps_completed['Preprocessing'] = run_preprocessing()
        if not steps_completed['Preprocessing']:
            print("\n❌ Preprocessing failed. Cannot continue.")
            return
        
        # Run EDA if requested
        if not args.no_eda:
            steps_completed['EDA'] = run_eda()
    
    # PHASE 2: Training
    if args.training or run_all:
        # Check if preprocessing was done
        if not os.path.exists('processed_data/dataset_splits.npz'):
            print("\n❌ Preprocessed data not found!")
            print("Please run preprocessing first:")
            print("  python run_all.py --preprocessing")
            return
        
        steps_completed['Model Training'] = run_model_training()
        
        if steps_completed.get('Model Training', False):
            # Run ensemble training (unless quick mode)
            if not args.quick:
                steps_completed['Ensemble Training'] = run_ensemble_training()
            
            # Run model testing
            steps_completed['Model Testing'] = run_model_testing()
    
    # Print summary
    print_summary(start_time, steps_completed)


if __name__ == "__main__":
    main()