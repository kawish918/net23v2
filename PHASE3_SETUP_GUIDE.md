# 🚀 Phase 3: Real-Time IDS Setup Guide

## 📁 Updated Folder Structure

```
net_23v2/
│
├── TII-SSRC-23/               # Original dataset
│   └── csv/
│       └── data.csv
│
├── processed_data/            # Preprocessed data (Phase 1)
│   ├── dataset_splits.npz
│   ├── scaler.pkl            ⭐ USED IN PHASE 3
│   ├── attack_mapping.json   ⭐ USED IN PHASE 3
│   ├── feature_mapping.json
│   └── ...
│
├── models/                    # Trained models (Phase 2)
│   ├── tii_ssrc23_best_model_v1.pkl  ⭐ USED IN PHASE 3 (LightGBM)
│   ├── checkpoint_RandomForest.pkl
│   ├── checkpoint_LightGBM.pkl
│   └── checkpoint_GradientBoosting.pkl
│
├── results/                   # Training results (Phase 2)
│   ├── training_summary.json
│   ├── model_comparison.csv
│   └── plots/
│
├── logs/                      # ⭐ NEW - Detection logs (Phase 3)
│   └── detections.csv        # Real-time detection log
│
├── src/
│   ├── preprocessing.py       # Phase 1
│   ├── train_model.py         # Phase 2
│   ├── realtime_ids.py        # ⭐ NEW - Phase 3 (Main IDS)
│   └── ...
│
├── utils/
│   └── feature_mapping.py
│
├── venv/                      # Virtual environment
│
├── requirements.txt           # Original dependencies
├── requirements_phase3.txt    # ⭐ NEW - Phase 3 dependencies
│
├── README.md
├── PHASE1_README.md
├── PHASE_2_README.md
└── PHASE3_SETUP_GUIDE.md      # ⭐ THIS FILE

```

---

## 🖥️ PC Setup Requirements

### **PC 1: Windows (IDS/Defender)**
**What you already have:**
- ✅ Python 3.12.4
- ✅ Virtual environment (`venv/`)
- ✅ Trained LightGBM model (99.9992% accuracy)
- ✅ Scaler and preprocessing objects

**What you need to install:**

#### 1️⃣ Npcap (Packet Capture Driver)
```
Download: https://npcap.com/#download
Installation:
1. Run the installer as Administrator
2. ✓ Check "Install Npcap in WinPcap API-compatible Mode"
3. ✓ Check "Support raw 802.11 traffic"
4. Install and reboot if prompted
```

#### 2️⃣ Python Dependencies
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install Phase 3 dependencies
pip install -r requirements_phase3.txt

# This installs:
# - scapy (packet capture & manipulation)
# - pyshark (alternative packet analyzer)
# - netifaces (network interface info)
```

#### 3️⃣ Network Configuration
```powershell
# Find your IP address
ipconfig

# Look for "IPv4 Address" under your active adapter
# Example: 192.168.1.100
# Write this down - you'll use it from Kali

# Optional: Disable Windows Firewall for testing
# Settings -> Windows Security -> Firewall -> Turn off (temporarily)
```

---

### **PC 2: Kali Linux (Attacker)**
**What you already have:**
- ✅ Most attack tools pre-installed

**What you need:**
```bash
# Update package list
sudo apt update

# Install any missing tools
sudo apt install -y hping3 nmap hydra slowhttptest

# Verify installations
hping3 --version
nmap --version
hydra -h
slowhttptest -h
```

**Network Configuration:**
```bash
# Find your IP address
ip addr show

# Test connectivity to Windows PC
ping <WINDOWS_IP>

# Example: ping 192.168.1.100
# Should get replies - if not, check network/firewall
```

---

## 🚀 How to Run Phase 3

### **Step 1: Start IDS on Windows**

```powershell
# Open PowerShell as ADMINISTRATOR (required for packet capture)
# Right-click PowerShell -> "Run as Administrator"

# Navigate to project
cd C:\Users\kawis\Desktop\Py\net_23v2

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the IDS
python src/realtime_ids.py
```

**Expected Output:**
```
================================================================================
SENTINET - REAL-TIME INTRUSION DETECTION SYSTEM
================================================================================

[+] Loading ML model and preprocessing objects...
✓ Model loaded: tii_ssrc23_best_model_v1.pkl
✓ Scaler loaded: scaler.pkl
✓ Attack mapping: {0: 'Benign', 1: 'Malicious'}
✓ Detection log: logs/detections.csv

[+] IDS initialized successfully!
================================================================================

[+] Starting packet capture...
    Interface: All interfaces
    Packet limit: Unlimited

[!] Press Ctrl+C to stop capture

================================================================================
```

---

### **Step 2: Generate Attack Traffic from Kali**

Open terminal on Kali and run these attacks:

#### **Attack 1: SYN Flood (DDoS)**
```bash
sudo hping3 -S -p 80 --flood 192.168.1.100

# -S: SYN flag
# -p 80: Target port 80 (HTTP)
# --flood: Send packets as fast as possible
# Replace 192.168.1.100 with your Windows IP
```

#### **Attack 2: Port Scan**
```bash
sudo nmap -sS -p 1-1000 192.168.1.100

# -sS: SYN stealth scan
# -p 1-1000: Scan first 1000 ports
```

#### **Attack 3: ICMP Flood**
```bash
sudo hping3 --icmp --flood 192.168.1.100

# --icmp: ICMP packets (ping flood)
# --flood: Maximum speed
```

#### **Attack 4: Slowloris (HTTP DoS)**
```bash
# Only if Windows has a web server running
slowhttptest -c 1000 -H -g -o slowloris -i 10 -r 200 -t GET -u http://192.168.1.100

# -c 1000: 1000 connections
# -H: Slowloris mode
# -i 10: 10 second intervals
```

#### **Attack 5: UDP Flood**
```bash
sudo hping3 --udp -p 53 --flood 192.168.1.100

# --udp: UDP packets
# -p 53: Target DNS port
```

---

### **Step 3: Monitor Detections on Windows**

The IDS will display alerts in real-time:

```
================================================================================
🚨 ALERT: MALICIOUS TRAFFIC DETECTED!
================================================================================
Timestamp:    2025-10-22 14:35:22
Source:       192.168.1.50:43521
Destination:  192.168.1.100:80
Protocol:     TCP
Confidence:   99.9977%
Flow Stats:   156 packets, 12480 bytes
Probabilities: Benign=0.02%, Malicious=99.98%
================================================================================
```

**Statistics (displayed every 30 seconds):**
```
================================================================================
REAL-TIME STATISTICS
================================================================================
Runtime:          120.5 seconds
Total Packets:    15,432
Total Flows:      87
Active Flows:     12
Packet Rate:      128.1 packets/sec
Benign Traffic:   72
Malicious Traffic: 15
Detection Rate:   17.24%
================================================================================
```

---

### **Step 4: Stop and Review Logs**

**Stop the IDS:**
```
Press Ctrl+C in the PowerShell window
```

**View Detection Log:**
```powershell
# Open the CSV log
notepad logs/detections.csv

# Or view in terminal
Get-Content logs/detections.csv | Select-Object -Last 20
```

**Log Format:**
```csv
timestamp,src_ip,src_port,dst_ip,dst_port,protocol,prediction,confidence,flow_packets,flow_bytes
2025-10-22 14:35:22,192.168.1.50,43521,192.168.1.100,80,TCP,Malicious,99.9977,156,12480
2025-10-22 14:35:28,192.168.1.100,80,8.8.8.8,443,TCP,Benign,99.9975,42,8520
...
```

---

## 🔧 Troubleshooting

### **Problem 1: "Permission Denied" or "Access Denied"**
**Solution:**
- Run PowerShell as Administrator
- Right-click PowerShell icon -> "Run as Administrator"

### **Problem 2: "Npcap not found" or "WinPcap not found"**
**Solution:**
- Install Npcap: https://npcap.com/#download
- During installation, enable "WinPcap API-compatible Mode"
- Reboot Windows

### **Problem 3: No packets captured**
**Solution:**
```powershell
# List available network interfaces
python -c "from scapy.all import get_if_list; print(get_if_list())"

# Specify interface manually in realtime_ids.py
# Edit src/realtime_ids.py, line 344:
ids.start_capture(interface='Wi-Fi', packet_count=0)  # Replace 'Wi-Fi' with your interface
```

### **Problem 4: Can't ping Windows from Kali**
**Solution:**
- Disable Windows Firewall temporarily
- Check both PCs are on same network (same subnet)
- Verify IPs with `ipconfig` (Windows) and `ip addr` (Kali)

### **Problem 5: Model detects all traffic as malicious**
**Solution:**
- This is expected if you're ONLY running attacks from Kali
- For benign traffic, browse the web, stream video, or SSH normally
- The IDS will classify normal traffic as Benign

### **Problem 6: Import errors**
**Solution:**
```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements_phase3.txt
```

---

## 📊 Understanding the Results

### **High Confidence = Good**
- **99%+ confidence**: Model is very certain
- **50-70% confidence**: Borderline case, manual review recommended

### **Detection Rate**
- **Depends on your network traffic**
- If only running attacks: High detection rate (50-100%)
- If normal browsing + attacks: Lower rate (5-20%)

### **False Positives**
- Should be **very rare** (model has 99.9992% accuracy)
- If you see many false positives, check feature extraction

---

## 🎯 Testing Checklist

- [ ] Npcap installed on Windows
- [ ] Virtual environment activated
- [ ] Phase 3 dependencies installed (`pip install -r requirements_phase3.txt`)
- [ ] IDS running as Administrator
- [ ] Kali can ping Windows IP
- [ ] SYN flood detected
- [ ] Port scan detected
- [ ] ICMP flood detected
- [ ] Normal traffic classified as Benign
- [ ] Logs saved to `logs/detections.csv`

---

## 📈 Next Steps (Phase 4)

Once Phase 3 is working:
1. **Desktop UI** - Build PySide6 GUI for real-time dashboard
2. **Alert System** - Email/SMS notifications for critical alerts
3. **Database Integration** - Store detections in SQLite/PostgreSQL
4. **Advanced Features** - Packet replay, traffic visualization, geolocation

---

## 🆘 Need Help?

**Common Issues:**
- Packet capture not working → Run as Administrator
- No Npcap → Download from https://npcap.com/#download
- Import errors → Activate venv and install requirements_phase3.txt
- Can't detect attacks → Check Kali can ping Windows IP

**Check Your Setup:**
```powershell
# Verify Python
python --version  # Should be 3.12.4

# Verify venv is activated
where python  # Should point to venv\Scripts\python.exe

# Verify Scapy can capture
python -c "from scapy.all import sniff; print('Scapy OK')"

# Verify model exists
Test-Path models/tii_ssrc23_best_model_v1.pkl  # Should return True
```

---

**Ready to test? Run the IDS on Windows, launch attacks from Kali, and watch the detections roll in! 🚀**
