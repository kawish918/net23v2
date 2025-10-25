# 🚀 GitHub Upload & Clone Guide for SentiNet IDS

## 📦 What to Upload to GitHub

### ✅ Files to Include:
```
sentinet-ids/
│
├── models/                              # ✅ INCLUDE (6.99 MB - under GitHub limit)
│   ├── tii_ssrc23_best_model_v1.pkl
│   ├── checkpoint_RandomForest.pkl
│   ├── checkpoint_LightGBM.pkl
│   └── checkpoint_GradientBoosting.pkl
│
├── processed_data/                      # ✅ INCLUDE (preprocessing objects)
│   ├── scaler.pkl
│   ├── attack_mapping.json
│   ├── feature_mapping.json
│   ├── top_features.json
│   ├── preprocessing_report.json
│   └── high_correlations.txt
│   # ❌ EXCLUDE: dataset_splits.npz (too large)
│
├── src/                                 # ✅ INCLUDE (all Python scripts)
│   ├── gui_ids.py                      # Main GUI application
│   ├── realtime_ids.py                 # Terminal IDS
│   ├── train_model.py
│   ├── preprocessing.py
│   ├── test_trained_model.py
│   └── ...
│
├── results/                             # ✅ INCLUDE (training results)
│   ├── training_summary.json
│   ├── model_comparison.csv
│   └── plots/                          # ✅ Keep structure
│       └── .gitkeep
│
├── logs/                                # ✅ INCLUDE (empty folder structure)
│   └── .gitkeep
│
├── utils/                               # ✅ INCLUDE
│   └── feature_mapping.py
│
├── .gitignore                           # ✅ INCLUDE
├── requirements.txt                     # ✅ INCLUDE
├── README.md                            # ✅ INCLUDE
├── PHASE3_SETUP_GUIDE.md               # ✅ INCLUDE
├── PHASE_2_README.md                   # ✅ INCLUDE (optional)
├── PHASE1_README.md                    # ✅ INCLUDE (optional)
└── GITHUB_SETUP.md                     # ✅ INCLUDE (this file)
```

### ❌ Files to EXCLUDE:
```
❌ TII-SSRC-23/csv/data.csv         # Original dataset (too large)
❌ processed_data/dataset_splits.npz # Preprocessed splits (too large)
❌ venv/                             # Virtual environment (user creates own)
❌ __pycache__/                      # Python cache
❌ logs/*.csv                        # User-generated logs
❌ *.pyc, *.pyo                      # Compiled Python
```

---

## 🔧 Step 1: Prepare Repository

### Delete files NOT needed on GitHub:
```powershell
# Navigate to project
cd C:\Users\kawis\Desktop\Py\net_23v2

# Remove large dataset file (if exists)
Remove-Item -Path "TII-SSRC-23\csv\data.csv" -ErrorAction SilentlyContinue

# Remove large processed splits
Remove-Item -Path "processed_data\dataset_splits.npz" -ErrorAction SilentlyContinue

# Remove venv (users will create their own)
# DON'T DELETE YET - you need it for testing!
# Remove-Item -Path "venv" -Recurse -Force
```

**Note:** Keep `venv/` on your local system for now. `.gitignore` will prevent it from uploading.

---

## 🌐 Step 2: Create GitHub Repository

### Option A: GitHub Website
1. Go to [github.com](https://github.com)
2. Click "+" → "New repository"
3. Repository name: `sentinet-ids`
4. Description: `Real-Time Network Intrusion Detection System with 99.9992% accuracy`
5. **Public** (or Private if you prefer)
6. ❌ **DON'T** initialize with README (you already have one)
7. Click "Create repository"

### Option B: GitHub CLI
```bash
# Install GitHub CLI: https://cli.github.com/
gh repo create sentinet-ids --public --source=. --remote=origin
```

---

## 📤 Step 3: Upload to GitHub

```powershell
# Navigate to project
cd C:\Users\kawis\Desktop\Py\net_23v2

# Initialize Git (if not already)
git init

# Add all files (respects .gitignore)
git add .

# Check what will be uploaded
git status

# Commit
git commit -m "Initial commit: SentiNet IDS v1.0 with pre-trained models"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/sentinet-ids.git

# Push to GitHub
git push -u origin master
```

**Alternative branch name (if using 'main'):**
```powershell
git branch -M main
git push -u origin main
```

---

## 📥 Step 4: Clone on Another System

### On the target Windows PC:

```powershell
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/sentinet-ids.git
cd sentinet-ids

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install Npcap (if not already installed)
# Download: https://npcap.com/#download
# Enable "WinPcap API-compatible Mode"

# 6. Run GUI (as Administrator)
python src/gui_ids.py
```

---

## ✅ Verification Checklist

Before uploading to GitHub, verify:

- [ ] `.gitignore` file exists and excludes `venv/`, `*.npz`, `data.csv`
- [ ] `models/` folder contains all 4 `.pkl` files
- [ ] `processed_data/` contains `scaler.pkl` and `attack_mapping.json`
- [ ] `src/gui_ids.py` exists and is executable
- [ ] `requirements.txt` includes Scapy and PySide6
- [ ] `README.md` has clear setup instructions
- [ ] `logs/.gitkeep` exists (keeps empty folder in git)
- [ ] Total size < 100MB (models are ~7MB, safe to upload)

**Check total size:**
```powershell
# Get total size of files to upload
Get-ChildItem -Recurse -File | Where-Object { 
    $_.FullName -notmatch "venv|__pycache__|\.git|data\.csv|\.npz" 
} | Measure-Object -Property Length -Sum | Select-Object @{Name="Size(MB)";Expression={[math]::Round($_.Sum/1MB,2)}}
```

---

## 🔍 What Users Get After Cloning

### Included (Ready to Use):
✅ **Pre-trained LightGBM model** (99.9992% accuracy)  
✅ **Preprocessing objects** (scaler, mappings)  
✅ **GUI application** (`gui_ids.py`)  
✅ **All source code**  
✅ **Training results** (metrics, plots)  
✅ **Complete documentation**

### NOT Included (Too Large):
❌ Original dataset (`data.csv` - 8.6M samples)  
❌ Preprocessed splits (`dataset_splits.npz`)  
❌ Virtual environment (user creates their own)

**Users can:**
- ✅ **Run the IDS immediately** (no training needed)
- ✅ **Test with Kali attacks** (SYN flood, port scan, etc.)
- ✅ **View real-time detections** in GUI
- ❌ **Cannot retrain models** without original dataset

---

## 🎯 Testing After Clone (Another System)

### Setup on Fresh Windows PC:

```powershell
# Prerequisites
# 1. Python 3.12+ installed
# 2. Npcap installed (https://npcap.com/#download)
# 3. Git installed

# Clone
git clone https://github.com/YOUR_USERNAME/sentinet-ids.git
cd sentinet-ids

# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Verify models exist
Test-Path models/tii_ssrc23_best_model_v1.pkl  # Should return True

# Run GUI (as Administrator)
python src/gui_ids.py
```

### Test Detection with Kali:

```bash
# On Kali Linux
# Get Windows IP: ipconfig on Windows

# Launch SYN flood
sudo hping3 -S -p 80 --flood 192.168.1.100

# Watch GUI for alerts! 🚨
```

---

## 📊 Repository Statistics

**Expected Stats:**
- **Files:** ~50-60 files
- **Size:** ~10-15 MB (models + code)
- **Languages:** Python (100%)
- **Dependencies:** 15+ packages

**GitHub will show:**
```
Languages:
  Python    98.5%
  Markdown   1.5%
```

---

## 🐛 Common Issues After Clone

### Issue 1: "Model not found"
**Cause:** Models not downloaded from GitHub  
**Solution:** 
```powershell
# Verify models exist
Get-ChildItem models/

# If missing, re-clone or check .gitignore
```

### Issue 2: "Module not found: scapy"
**Cause:** Dependencies not installed  
**Solution:**
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue 3: "Npcap not found"
**Cause:** Npcap not installed on new system  
**Solution:** Download and install [Npcap](https://npcap.com/#download)

### Issue 4: "Permission denied"
**Cause:** Not running as Administrator  
**Solution:** Right-click PowerShell → "Run as Administrator"

---

## 📈 Optional: Add GitHub Features

### Add Topics (on GitHub website):
```
machine-learning, intrusion-detection, cybersecurity, 
lightgbm, python, pyside6, network-security, ids, 
real-time-detection, scapy
```

### Add Description:
```
Real-time network intrusion detection system with 99.9992% accuracy 
using LightGBM and Scapy. Features modern PySide6 GUI for live 
threat monitoring.
```

### Add License:
- Go to repository → "Add file" → "Create new file"
- Name: `LICENSE`
- Choose template: "MIT License"

---

## 🎉 Success Criteria

After uploading, verify on GitHub:

1. [ ] Repository is public/accessible
2. [ ] README.md displays correctly with formatting
3. [ ] `models/` folder contains 4 `.pkl` files
4. [ ] `src/gui_ids.py` is visible
5. [ ] `.gitignore` is present
6. [ ] Total size shows ~10-15 MB
7. [ ] No `venv/` folder visible
8. [ ] No `data.csv` visible

---

## 🚀 Next Steps

**After successful upload:**

1. **Clone on test system** (another Windows PC)
2. **Follow README.md** instructions
3. **Run GUI** (`python src/gui_ids.py`)
4. **Test with Kali attacks** (SYN flood, port scan)
5. **Verify detections** appear in GUI
6. **Check logs** (`logs/detections.csv`)

**Share your repository:**
```
🔗 https://github.com/YOUR_USERNAME/sentinet-ids

⭐ Star the repo
🍴 Fork for your own experiments
🐛 Report issues
🤝 Contribute improvements
```

---

**Ready to upload? Run the commands in Step 3! 🚀**
