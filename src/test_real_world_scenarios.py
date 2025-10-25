"""
SentiNet - Real-World Attack Scenario Testing
Tests trained models against realistic network attack patterns

This script creates hardcoded examples of:
- Normal traffic patterns
- Various attack types (DDoS, Port Scan, Brute Force, etc.)
- Edge cases and adversarial examples
- Real-world network behaviors

Author: SentiNet Development Team
"""

import numpy as np
import joblib
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RealWorldScenarioTester:
    """
    Comprehensive testing with realistic network attack scenarios
    """
    
    def __init__(self, model_path='models/tii_ssrc23_best_model_v1.pkl',
                 scaler_path='processed_data/scaler.pkl',
                 mapping_path='processed_data/attack_mapping.json'):
        """
        Initialize tester with trained model
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            mapping_path: Path to attack label mapping
        """
        print("="*80)
        print("REAL-WORLD SCENARIO TESTING SYSTEM")
        print("="*80)
        
        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("✓ Model loaded")
        
        # Load scaler
        print(f"Loading scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)
        print("✓ Scaler loaded")
        
        # Load attack mapping
        with open(mapping_path, 'r') as f:
            self.attack_mapping = json.load(f)
        self.label_to_attack = {int(v): k for k, v in self.attack_mapping.items()}
        print("✓ Attack mapping loaded")
        
        # Feature names (79 features in order)
        self.feature_names = [
            'Src Port', 'Dst Port', 'Protocol', 'Flow Duration',
            'Total Fwd Packet', 'Total Bwd packets',
            'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
            'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
            'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
            'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags',
            'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s',
            'Packet Length Min', 'Packet Length Max',
            'Packet Length Mean', 'Packet Length Std',
            'Packet Length Variance',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWR Flag Count', 'ECE Flag Count',
            'Down/Up Ratio', 'Average Packet Size',
            'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
            'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
            'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'FWD Init Win Bytes', 'Bwd Init Win Bytes',
            'Fwd Act Data Pkts', 'Fwd Seg Size Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        
        print(f"✓ Testing with {len(self.feature_names)} features")
        print("\n" + "="*80)
    
    def create_normal_web_browsing(self):
        """Normal HTTPS web browsing - benign traffic"""
        return {
            'name': 'Normal Web Browsing (HTTPS)',
            'expected': 'Benign',
            'description': 'User browsing websites over HTTPS',
            'features': np.array([[
                52341,      # Src Port (random high port)
                443,        # Dst Port (HTTPS)
                6,          # Protocol (TCP)
                5234.5,     # Flow Duration (5.2 seconds)
                45,         # Total Fwd Packet
                42,         # Total Bwd packets
                12850,      # Total Length of Fwd Packet
                78432,      # Total Length of Bwd Packet (server sends more)
                1460,       # Fwd Packet Length Max (MTU size)
                52,         # Fwd Packet Length Min
                285.5,      # Fwd Packet Length Mean
                324.2,      # Fwd Packet Length Std
                1460,       # Bwd Packet Length Max
                52,         # Bwd Packet Length Min
                1867.4,     # Bwd Packet Length Mean (images, content)
                598.3,      # Bwd Packet Length Std
                2456.7,     # Flow Bytes/s
                16.6,       # Flow Packets/s
                60.2,       # Flow IAT Mean (reasonable)
                45.3,       # Flow IAT Std
                850.2,      # Flow IAT Max
                5.2,        # Flow IAT Min
                2800.3,     # Fwd IAT Total
                62.2,       # Fwd IAT Mean
                48.5,       # Fwd IAT Std
                890.2,      # Fwd IAT Max
                8.5,        # Fwd IAT Min
                2500.8,     # Bwd IAT Total
                59.5,       # Bwd IAT Mean
                42.1,       # Bwd IAT Std
                820.5,      # Bwd IAT Max
                6.8,        # Bwd IAT Min
                2,          # Fwd PSH Flags (some)
                3,          # Bwd PSH Flags (some)
                0,          # Fwd URG Flags
                0,          # Bwd URG Flags
                900,        # Fwd Header Length (20 bytes * 45 pkts)
                840,        # Bwd Header Length
                8.6,        # Fwd Packets/s
                8.0,        # Bwd Packets/s
                52,         # Packet Length Min
                1460,       # Packet Length Max
                1049.2,     # Packet Length Mean
                674.5,      # Packet Length Std
                454950.2,   # Packet Length Variance
                1,          # FIN Flag Count
                1,          # SYN Flag Count
                0,          # RST Flag Count
                5,          # PSH Flag Count
                85,         # ACK Flag Count (most packets)
                0,          # URG Flag Count
                0,          # CWR Flag Count
                0,          # ECE Flag Count
                0.93,       # Down/Up Ratio (more download)
                1049.2,     # Average Packet Size
                285.5,      # Fwd Segment Size Avg
                1867.4,     # Bwd Segment Size Avg
                0,          # Fwd Bytes/Bulk Avg
                0,          # Fwd Packet/Bulk Avg
                0,          # Fwd Bulk Rate Avg
                0,          # Bwd Bytes/Bulk Avg
                0,          # Bwd Packet/Bulk Avg
                0,          # Bwd Bulk Rate Avg
                45,         # Subflow Fwd Packets
                12850,      # Subflow Fwd Bytes
                42,         # Subflow Bwd Packets
                78432,      # Subflow Bwd Bytes
                29200,      # FWD Init Win Bytes (29KB - normal)
                28960,      # Bwd Init Win Bytes (28KB - normal)
                43,         # Fwd Act Data Pkts
                52,         # Fwd Seg Size Min
                1200.5,     # Active Mean
                456.2,      # Active Std
                2500.3,     # Active Max
                450.2,      # Active Min
                85.3,       # Idle Mean
                35.2,       # Idle Std
                250.5,      # Idle Max
                25.8        # Idle Min
            ]])
        }
    
    def create_ddos_attack(self):
        """DDoS flood attack - malicious traffic"""
        return {
            'name': 'DDoS Flood Attack',
            'expected': 'Malicious',
            'description': 'High-volume flood attack overwhelming server',
            'features': np.array([[
                54892,      # Src Port (spoofed/random)
                80,         # Dst Port (HTTP target)
                6,          # Protocol (TCP)
                0.5,        # Flow Duration (VERY SHORT - red flag)
                950,        # Total Fwd Packet (MASSIVE volume)
                0,          # Total Bwd packets (NO RESPONSE - red flag)
                47500,      # Total Length of Fwd Packet
                0,          # Total Length of Bwd Packet (NO RESPONSE)
                50,         # Fwd Packet Length Max (tiny packets)
                50,         # Fwd Packet Length Min (uniform size)
                50,         # Fwd Packet Length Mean (uniform)
                0.1,        # Fwd Packet Length Std (NO VARIATION - red flag)
                0,          # Bwd Packet Length Max
                0,          # Bwd Packet Length Min
                0,          # Bwd Packet Length Mean
                0,          # Bwd Packet Length Std
                95000000,   # Flow Bytes/s (EXTREMELY HIGH - red flag)
                1900000,    # Flow Packets/s (EXTREMELY HIGH - red flag)
                0.0005,     # Flow IAT Mean (TINY gaps - flood)
                0.0001,     # Flow IAT Std (uniform timing)
                0.001,      # Flow IAT Max
                0.0001,     # Flow IAT Min
                0.475,      # Fwd IAT Total (tiny)
                0.0005,     # Fwd IAT Mean
                0.0001,     # Fwd IAT Std
                0.001,      # Fwd IAT Max
                0.0001,     # Fwd IAT Min
                0,          # Bwd IAT Total
                0,          # Bwd IAT Mean
                0,          # Bwd IAT Std
                0,          # Bwd IAT Max
                0,          # Bwd IAT Min
                0,          # Fwd PSH Flags (NO FLAGS - raw flood)
                0,          # Bwd PSH Flags
                0,          # Fwd URG Flags
                0,          # Bwd URG Flags
                19000,      # Fwd Header Length (20 bytes * 950)
                0,          # Bwd Header Length
                1900000,    # Fwd Packets/s (MASSIVE)
                0,          # Bwd Packets/s
                50,         # Packet Length Min
                50,         # Packet Length Max (all same size)
                50,         # Packet Length Mean
                0.1,        # Packet Length Std (NO VARIATION)
                0.01,       # Packet Length Variance
                0,          # FIN Flag Count (NO PROPER CLOSING)
                950,        # SYN Flag Count (ALL SYN - SYN FLOOD)
                0,          # RST Flag Count
                0,          # PSH Flag Count
                0,          # ACK Flag Count (NO ACKS - incomplete handshake)
                0,          # URG Flag Count
                0,          # CWR Flag Count
                0,          # ECE Flag Count
                0,          # Down/Up Ratio (NO RESPONSE)
                50,         # Average Packet Size (tiny)
                50,         # Fwd Segment Size Avg
                0,          # Bwd Segment Size Avg
                0,          # Fwd Bytes/Bulk Avg
                0,          # Fwd Packet/Bulk Avg
                0,          # Fwd Bulk Rate Avg
                0,          # Bwd Bytes/Bulk Avg
                0,          # Bwd Packet/Bulk Avg
                0,          # Bwd Bulk Rate Avg
                950,        # Subflow Fwd Packets
                47500,      # Subflow Fwd Bytes
                0,          # Subflow Bwd Packets
                0,          # Subflow Bwd Bytes
                0,          # FWD Init Win Bytes (NO PROPER INIT)
                0,          # Bwd Init Win Bytes
                0,          # Fwd Act Data Pkts (NO DATA - just SYN)
                50,         # Fwd Seg Size Min
                0,          # Active Mean (NO ACTIVITY - just flood)
                0,          # Active Std
                0,          # Active Max
                0,          # Active Min
                0,          # Idle Mean
                0,          # Idle Std
                0,          # Idle Max
                0           # Idle Min
            ]])
        }
    
    def create_port_scan(self):
        """Port scanning attack - reconnaissance"""
        return {
            'name': 'Port Scan (Reconnaissance)',
            'expected': 'Malicious',
            'description': 'Attacker scanning for open ports',
            'features': np.array([[
                58234,      # Src Port (single source)
                22,         # Dst Port (SSH - one of many ports scanned)
                6,          # Protocol (TCP)
                0.05,       # Flow Duration (VERY SHORT - quick probe)
                1,          # Total Fwd Packet (SINGLE PACKET - red flag)
                0,          # Total Bwd packets (NO RESPONSE from closed port)
                54,         # Total Length of Fwd Packet (just SYN)
                0,          # Total Length of Bwd Packet
                54,         # Fwd Packet Length Max
                54,         # Fwd Packet Length Min
                54,         # Fwd Packet Length Mean
                0,          # Fwd Packet Length Std (single packet)
                0,          # Bwd Packet Length Max
                0,          # Bwd Packet Length Min
                0,          # Bwd Packet Length Mean
                0,          # Bwd Packet Length Std
                1080,       # Flow Bytes/s
                20,         # Flow Packets/s
                0,          # Flow IAT Mean (single packet)
                0,          # Flow IAT Std
                0,          # Flow IAT Max
                0,          # Flow IAT Min
                0,          # Fwd IAT Total
                0,          # Fwd IAT Mean
                0,          # Fwd IAT Std
                0,          # Fwd IAT Max
                0,          # Fwd IAT Min
                0,          # Bwd IAT Total
                0,          # Bwd IAT Mean
                0,          # Bwd IAT Std
                0,          # Bwd IAT Max
                0,          # Bwd IAT Min
                0,          # Fwd PSH Flags
                0,          # Bwd PSH Flags
                0,          # Fwd URG Flags
                0,          # Bwd URG Flags
                20,         # Fwd Header Length (single TCP header)
                0,          # Bwd Header Length
                20,         # Fwd Packets/s
                0,          # Bwd Packets/s
                54,         # Packet Length Min
                54,         # Packet Length Max
                54,         # Packet Length Mean
                0,          # Packet Length Std
                0,          # Packet Length Variance
                0,          # FIN Flag Count
                1,          # SYN Flag Count (JUST SYN - probe)
                0,          # RST Flag Count
                0,          # PSH Flag Count
                0,          # ACK Flag Count (NO ACK - incomplete)
                0,          # URG Flag Count
                0,          # CWR Flag Count
                0,          # ECE Flag Count
                0,          # Down/Up Ratio (NO RESPONSE)
                54,         # Average Packet Size
                54,         # Fwd Segment Size Avg
                0,          # Bwd Segment Size Avg
                0,          # Fwd Bytes/Bulk Avg
                0,          # Fwd Packet/Bulk Avg
                0,          # Fwd Bulk Rate Avg
                0,          # Bwd Bytes/Bulk Avg
                0,          # Bwd Packet/Bulk Avg
                0,          # Bwd Bulk Rate Avg
                1,          # Subflow Fwd Packets
                54,         # Subflow Fwd Bytes
                0,          # Subflow Bwd Packets
                0,          # Subflow Bwd Bytes
                1024,       # FWD Init Win Bytes (small window)
                0,          # Bwd Init Win Bytes
                0,          # Fwd Act Data Pkts (NO DATA)
                54,         # Fwd Seg Size Min
                0,          # Active Mean
                0,          # Active Std
                0,          # Active Max
                0,          # Active Min
                0,          # Idle Mean
                0,          # Idle Std
                0,          # Idle Max
                0           # Idle Min
            ]])
        }
    
    def create_ssh_brute_force(self):
        """SSH brute force attack - password guessing"""
        return {
            'name': 'SSH Brute Force Attack',
            'expected': 'Malicious',
            'description': 'Attacker trying many SSH login attempts',
            'features': np.array([[
                45892,      # Src Port
                22,         # Dst Port (SSH)
                6,          # Protocol (TCP)
                120.5,      # Flow Duration (LONG - many attempts)
                456,        # Total Fwd Packet (MANY login attempts)
                445,        # Total Bwd packets (server responses)
                25680,      # Total Length of Fwd Packet
                24560,      # Total Length of Bwd Packet
                80,         # Fwd Packet Length Max (small - just credentials)
                52,         # Fwd Packet Length Min
                56.3,       # Fwd Packet Length Mean (small packets)
                12.5,       # Fwd Packet Length Std
                80,         # Bwd Packet Length Max
                52,         # Bwd Packet Length Min
                55.2,       # Bwd Packet Length Mean
                11.8,       # Bwd Packet Length Std
                416.5,      # Flow Bytes/s
                7.5,        # Flow Packets/s
                264.5,      # Flow IAT Mean (REGULAR TIMING - automated)
                15.2,       # Flow IAT Std (LOW VARIATION - bot)
                350.5,      # Flow IAT Max
                200.3,      # Flow IAT Min
                120230.5,   # Fwd IAT Total
                263.8,      # Fwd IAT Mean (RHYTHMIC - bot behavior)
                14.5,       # Fwd IAT Std
                345.2,      # Fwd IAT Max
                205.3,      # Fwd IAT Min
                119890.2,   # Bwd IAT Total
                269.3,      # Bwd IAT Mean
                16.8,       # Bwd IAT Std
                355.8,      # Bwd IAT Max
                198.5,      # Bwd IAT Min
                456,        # Fwd PSH Flags (ALL packets push data)
                445,        # Bwd PSH Flags
                0,          # Fwd URG Flags
                0,          # Bwd URG Flags
                9120,       # Fwd Header Length
                8900,       # Bwd Header Length
                3.8,        # Fwd Packets/s
                3.7,        # Bwd Packets/s
                52,         # Packet Length Min
                80,         # Packet Length Max
                55.7,       # Packet Length Mean
                12.1,       # Packet Length Std
                146.4,      # Packet Length Variance
                1,          # FIN Flag Count (eventually closes)
                1,          # SYN Flag Count (initial connect)
                0,          # RST Flag Count
                901,        # PSH Flag Count (HIGH - constant data)
                898,        # ACK Flag Count
                0,          # URG Flag Count
                0,          # CWR Flag Count
                0,          # ECE Flag Count
                0.96,       # Down/Up Ratio (balanced)
                55.7,       # Average Packet Size
                56.3,       # Fwd Segment Size Avg
                55.2,       # Bwd Segment Size Avg
                0,          # Fwd Bytes/Bulk Avg
                0,          # Fwd Packet/Bulk Avg
                0,          # Fwd Bulk Rate Avg
                0,          # Bwd Bytes/Bulk Avg
                0,          # Bwd Packet/Bulk Avg
                0,          # Bwd Bulk Rate Avg
                456,        # Subflow Fwd Packets
                25680,      # Subflow Fwd Bytes
                445,        # Subflow Bwd Packets
                24560,      # Subflow Bwd Bytes
                5840,       # FWD Init Win Bytes
                5840,       # Bwd Init Win Bytes
                454,        # Fwd Act Data Pkts
                52,         # Fwd Seg Size Min
                2500.5,     # Active Mean (LONG active periods)
                1200.3,     # Active Std
                5000.2,     # Active Max
                1000.5,     # Active Min
                5.2,        # Idle Mean (SHORT idle - persistent)
                2.5,        # Idle Std
                15.3,       # Idle Max
                1.2         # Idle Min
            ]])
        }
    
    def create_normal_ssh_session(self):
        """Normal SSH session - legitimate remote work"""
        return {
            'name': 'Normal SSH Session (Legitimate)',
            'expected': 'Benign',
            'description': 'System administrator working remotely',
            'features': np.array([[
                51234,      # Src Port
                22,         # Dst Port (SSH)
                6,          # Protocol (TCP)
                1850.5,     # Flow Duration (30+ minutes - real work)
                2450,       # Total Fwd Packet (typing, commands)
                2380,       # Total Bwd packets (terminal output)
                185400,     # Total Length of Fwd Packet
                892340,     # Total Length of Bwd Packet (more output than input)
                1460,       # Fwd Packet Length Max
                52,         # Fwd Packet Length Min
                75.7,       # Fwd Packet Length Mean (varied - human typing)
                245.3,      # Fwd Packet Length Std (HIGH VARIATION - human)
                1460,       # Bwd Packet Length Max
                52,         # Bwd Packet Length Min
                374.9,      # Bwd Packet Length Mean
                458.2,      # Bwd Packet Length Std (varied responses)
                582.1,      # Flow Bytes/s
                2.6,        # Flow Packets/s
                755.3,      # Flow IAT Mean (IRREGULAR - human typing)
                2345.8,     # Flow IAT Std (HIGH VARIATION - pauses, thinking)
                45000.5,    # Flow IAT Max (long pause - thinking/reading)
                5.2,        # Flow IAT Min (quick keystrokes)
                1850245.3,  # Fwd IAT Total
                755.4,      # Fwd IAT Mean
                2346.2,     # Fwd IAT Std (HUMAN BEHAVIOR)
                45005.3,    # Fwd IAT Max
                5.5,        # Fwd IAT Min
                1849890.5,  # Bwd IAT Total
                777.3,      # Bwd IAT Mean
                2298.5,     # Bwd IAT Std
                44850.2,    # Bwd IAT Max
                6.2,        # Bwd IAT Min
                1200,       # Fwd PSH Flags (normal)
                1180,       # Bwd PSH Flags
                0,          # Fwd URG Flags
                0,          # Bwd URG Flags
                49000,      # Fwd Header Length
                47600,      # Bwd Header Length
                1.3,        # Fwd Packets/s
                1.3,        # Bwd Packets/s
                52,         # Packet Length Min
                1460,       # Packet Length Max
                223.1,      # Packet Length Mean
                395.8,      # Packet Length Std
                156657.6,   # Packet Length Variance
                1,          # FIN Flag Count
                1,          # SYN Flag Count
                0,          # RST Flag Count
                2380,       # PSH Flag Count
                4828,       # ACK Flag Count
                0,          # URG Flag Count
                0,          # CWR Flag Count
                0,          # ECE Flag Count
                4.81,       # Down/Up Ratio (more output than input)
                223.1,      # Average Packet Size
                75.7,       # Fwd Segment Size Avg
                374.9,      # Bwd Segment Size Avg
                0,          # Fwd Bytes/Bulk Avg
                0,          # Fwd Packet/Bulk Avg
                0,          # Fwd Bulk Rate Avg
                0,          # Bwd Bytes/Bulk Avg
                0,          # Bwd Packet/Bulk Avg
                0,          # Bwd Bulk Rate Avg
                2450,       # Subflow Fwd Packets
                185400,     # Subflow Fwd Bytes
                2380,       # Subflow Bwd Packets
                892340,     # Subflow Bwd Bytes
                29200,      # FWD Init Win Bytes (normal)
                29200,      # Bwd Init Win Bytes
                2448,       # Fwd Act Data Pkts
                52,         # Fwd Seg Size Min
                15000.5,    # Active Mean (long active periods)
                8500.3,     # Active Std
                45000.2,    # Active Max
                2000.5,     # Active Min
                5000.3,     # Idle Mean (occasional pauses)
                3500.2,     # Idle Std
                25000.5,    # Idle Max (thinking time)
                500.2       # Idle Min
            ]])
        }
    
    def create_dns_tunneling(self):
        """DNS tunneling attack - data exfiltration"""
        return {
            'name': 'DNS Tunneling (Data Exfiltration)',
            'expected': 'Malicious',
            'description': 'Attacker using DNS to steal data',
            'features': np.array([[
                53124,      # Src Port (DNS client)
                53,         # Dst Port (DNS)
                17,         # Protocol (UDP - DNS uses UDP)
                45.5,       # Flow Duration
                890,        # Total Fwd Packet (MANY DNS queries - red flag)
                890,        # Total Bwd packets (responses)
                67200,      # Total Length of Fwd Packet (large for DNS)
                26700,      # Total Length of Bwd Packet
                120,        # Fwd Packet Length Max (LARGE DNS query - tunneling)
                60,         # Fwd Packet Length Min
                75.5,       # Fwd Packet Length Mean (larger than normal DNS)
                18.3,       # Fwd Packet Length Std
                60,         # Bwd Packet Length Max (normal DNS response)
                30,         # Bwd Packet Length Min
                30,         # Bwd Packet Length Mean
                5.2,        # Bwd Packet Length Std
                2063.7,     # Flow Bytes/s
                39.1,       # Flow Packets/s (HIGH RATE - automated)
                51.1,       # Flow IAT Mean (REGULAR - automated)
                5.2,        # Flow IAT Std (LOW VARIATION - bot)
                85.3,       # Flow IAT Max
                35.2,       # Flow IAT Min
                45489.5,    # Fwd IAT Total
                51.1,       # Fwd IAT Mean (RHYTHMIC)
                5.1,        # Fwd IAT Std
                84.5,       # Fwd IAT Max
                36.2,       # Fwd IAT Min
                45445.8,    # Bwd IAT Total
                51.0,       # Bwd IAT Mean
                5.3,        # Bwd IAT Std
                86.2,       # Bwd IAT Max
                35.8,       # Bwd IAT Min
                0,          # Fwd PSH Flags (UDP - no flags)
                0,          # Bwd PSH Flags
                0,          # Fwd URG Flags
                0,          # Bwd URG Flags
                7120,       # Fwd Header Length (UDP headers)
                7120,       # Bwd Header Length
                19.6,       # Fwd Packets/s (HIGH for DNS)
                19.6,       # Bwd Packets/s
                30,         # Packet Length Min
                120,        # Packet Length Max
                52.8,       # Packet Length Mean
                28.5,       # Packet Length Std
                812.2,      # Packet Length Variance
                0,          # FIN Flag Count (UDP)
                0,          # SYN Flag Count (UDP)
                0,          # RST Flag Count
                0,          # PSH Flag Count
                0,          # ACK Flag Count
                0,          # URG Flag Count
                0,          # CWR Flag Count
                0,          # ECE Flag Count
                0.40,       # Down/Up Ratio (more queries than responses - unusual)
                52.8,       # Average Packet Size
                75.5,       # Fwd Segment Size Avg
                30,         # Bwd Segment Size Avg
                0,          # Fwd Bytes/Bulk Avg
                0,          # Fwd Packet/Bulk Avg
                0,          # Fwd Bulk Rate Avg
                0,          # Bwd Bytes/Bulk Avg
                0,          # Bwd Packet/Bulk Avg
                0,          # Bwd Bulk Rate Avg
                890,        # Subflow Fwd Packets
                67200,      # Subflow Fwd Bytes
                890,        # Subflow Bwd Packets
                26700,      # Subflow Bwd Bytes
                0,          # FWD Init Win Bytes (UDP)
                0,          # Bwd Init Win Bytes
                890,        # Fwd Act Data Pkts
                60,         # Fwd Seg Size Min
                500.3,      # Active Mean
                150.2,      # Active Std
                1200.5,     # Active Max
                200.3,      # Active Min
                50.2,       # Idle Mean
                20.5,       # Idle Std
                150.3,      # Idle Max
                15.2        # Idle Min
            ]])
        }
    
    def create_normal_dns_queries(self):
        """Normal DNS queries - legitimate name resolution"""
        return {
            'name': 'Normal DNS Queries',
            'expected': 'Benign',
            'description': 'Legitimate DNS lookups for websites',
            'features': np.array([[
                54892,      # Src Port
                53,         # Dst Port (DNS)
                17,         # Protocol (UDP)
                2.5,        # Flow Duration (SHORT - quick lookups)
                3,          # Total Fwd Packet (few queries)
                3,          # Total Bwd packets
                180,        # Total Length of Fwd Packet (small)
                210,        # Total Length of Bwd Packet
                65,         # Fwd Packet Length Max
                55,         # Fwd Packet Length Min
                60,         # Fwd Packet Length Mean
                4.1,        # Fwd Packet Length Std
                80,         # Bwd Packet Length Max
                65,         # Bwd Packet Length Min
                70,         # Bwd Packet Length Mean
                6.1,        # Bwd Packet Length Std
                156,        # Flow Bytes/s
                2.4,        # Flow Packets/s (LOW - occasional)
                833.3,      # Flow IAT Mean (IRREGULAR - human browsing)
                456.2,      # Flow IAT Std (HIGH VARIATION)
                1500.5,     # Flow IAT Max
                250.2,      # Flow IAT Min
                2500,       # Fwd IAT Total
                1250,       # Fwd IAT Mean
                500.3,      # Fwd IAT Std
                1800.2,     # Fwd IAT Max
                700.5,      # Fwd IAT Min
                2450,       # Bwd IAT Total
                1225,       # Bwd IAT Mean
                480.5,      # Bwd IAT Std
                1750.3,     # Bwd IAT Max
                680.2,      # Bwd IAT Min
                0,          # Fwd PSH Flags
                0,          # Bwd PSH Flags
                0,          # Fwd URG Flags
                0,          # Bwd URG Flags
                24,         # Fwd Header Length (8 bytes * 3)
                24,         # Bwd Header Length
                1.2,        # Fwd Packets/s
                1.2,        # Bwd Packets/s
                55,         # Packet Length Min
                80,         # Packet Length Max
                65,         # Packet Length Mean
                8.5,        # Packet Length Std
                72.2,       # Packet Length Variance
                0,          # FIN Flag Count
                0,          # SYN Flag Count
                0,          # RST Flag Count
                0,          # PSH Flag Count
                0,          # ACK Flag Count
                0,          # URG Flag Count
                0,          # CWR Flag Count
                0,          # ECE Flag Count
                1.17,       # Down/Up Ratio (slightly more response)
                65,         # Average Packet Size
                60,         # Fwd Segment Size Avg
                70,         # Bwd Segment Size Avg
                0,          # Fwd Bytes/Bulk Avg
                0,          # Fwd Packet/Bulk Avg
                0,          # Fwd Bulk Rate Avg
                0,          # Bwd Bytes/Bulk Avg
                0,          # Bwd Packet/Bulk Avg
                0,          # Bwd Bulk Rate Avg
                3,          # Subflow Fwd Packets
                180,        # Subflow Fwd Bytes
                3,          # Subflow Bwd Packets
                210,        # Subflow Bwd Bytes
                0,          # FWD Init Win Bytes
                0,          # Bwd Init Win Bytes
                3,          # Fwd Act Data Pkts
                55,         # Fwd Seg Size Min
                850.5,      # Active Mean
                250.3,      # Active Std
                1200.5,     # Active Max
                500.2,      # Active Min
                850.2,      # Idle Mean
                250.5,      # Idle Std
                1200.8,     # Idle Max
                500.5       # Idle Min
            ]])
        }
    
    def create_slowloris_attack(self):
        """Slowloris attack - slow HTTP attack"""
        return {
            'name': 'Slowloris Attack',
            'expected': 'Malicious',
            'description': 'Slow HTTP attack keeping connections open',
            'features': np.array([[
                52341,      # Src Port
                80,         # Dst Port (HTTP)
                6,          # Protocol (TCP)
                300.5,      # Flow Duration (VERY LONG - keeps alive)
                150,        # Total Fwd Packet (many small packets)
                148,        # Total Bwd packets
                7500,       # Total Length of Fwd Packet (small total)
                7400,       # Total Length of Bwd Packet
                60,         # Fwd Packet Length Max (TINY packets)
                40,         # Fwd Packet Length Min
                50,         # Fwd Packet Length Mean (all small)
                5.2,        # Fwd Packet Length Std
                60,         # Bwd Packet Length Max
                40,         # Bwd Packet Length Min
                50,         # Bwd Packet Length Mean
                5.1,        # Bwd Packet Length Std
                49.7,       # Flow Bytes/s (VERY LOW - slow attack)
                0.99,       # Flow Packets/s (LOW rate)
                2003.3,     # Flow IAT Mean (LONG gaps - slowloris signature)
                500.5,      # Flow IAT Std
                3000.5,     # Flow IAT Max (VERY LONG - keeps connection)
                1500.2,     # Flow IAT Min
                300495.5,   # Fwd IAT Total
                2003.3,     # Fwd IAT Mean (SLOW sending)
                501.2,      # Fwd IAT Std
                3005.3,     # Fwd IAT Max
                1502.5,     # Fwd IAT Min
                299890.2,   # Bwd IAT Total
                2026.3,     # Bwd IAT Mean
                495.8,      # Bwd IAT Std
                2980.5,     # Bwd IAT Max
                1520.3,     # Bwd IAT Min
                150,        # Fwd PSH Flags (all packets)
                148,        # Bwd PSH Flags
                0,          # Fwd URG Flags
                0,          # Bwd URG Flags
                3000,       # Fwd Header Length
                2960,       # Bwd Header Length
                0.50,       # Fwd Packets/s (VERY LOW)
                0.49,       # Bwd Packets/s
                40,         # Packet Length Min
                60,         # Packet Length Max
                50,         # Packet Length Mean
                5.1,        # Packet Length Std
                26.0,       # Packet Length Variance
                0,          # FIN Flag Count (NO CLOSING - kept alive)
                1,          # SYN Flag Count
                0,          # RST Flag Count
                298,        # PSH Flag Count
                297,        # ACK Flag Count
                0,          # URG Flag Count
                0,          # CWR Flag Count
                0,          # ECE Flag Count
                0.99,       # Down/Up Ratio (balanced)
                50,         # Average Packet Size
                50,         # Fwd Segment Size Avg
                50,         # Bwd Segment Size Avg
                0,          # Fwd Bytes/Bulk Avg
                0,          # Fwd Packet/Bulk Avg
                0,          # Fwd Bulk Rate Avg
                0,          # Bwd Bytes/Bulk Avg
                0,          # Bwd Packet/Bulk Avg
                0,          # Bwd Bulk Rate Avg
                150,        # Subflow Fwd Packets
                7500,       # Subflow Fwd Bytes
                148,        # Subflow Bwd Packets
                7400,       # Subflow Bwd Bytes
                5840,       # FWD Init Win Bytes
                5840,       # Bwd Init Win Bytes
                149,        # Fwd Act Data Pkts
                40,         # Fwd Seg Size Min
                2000.5,     # Active Mean (LONG - stays active)
                500.2,      # Active Std
                3000.5,     # Active Max
                1500.2,     # Active Min
                5.2,        # Idle Mean (SHORT idle - keeps alive)
                2.5,        # Idle Std
                15.3,       # Idle Max
                1.2         # Idle Min
            ]])
        }
    
    def predict_scenario(self, scenario):
        """
        Predict on a single scenario and display detailed results
        
        Args:
            scenario: Dictionary with scenario details
        
        Returns:
            Dictionary with prediction results
        """
        # Scale features
        features_scaled = self.scaler.transform(scenario['features'])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get prediction name
        predicted_class = self.label_to_attack[prediction]
        
        # Get confidence
        confidence = probabilities[prediction] * 100
        
        # Check if correct
        is_correct = (predicted_class == scenario['expected'])
        
        return {
            'predicted': predicted_class,
            'expected': scenario['expected'],
            'confidence': confidence,
            'probabilities': {
                self.label_to_attack[i]: prob * 100 
                for i, prob in enumerate(probabilities)
            },
            'is_correct': is_correct
        }
    
    def run_all_tests(self):
        """Run all scenario tests and generate comprehensive report"""
        print("\n" + "="*80)
        print("RUNNING REAL-WORLD SCENARIO TESTS")
        print("="*80)
        
        # Define all scenarios
        scenarios = [
            self.create_normal_web_browsing(),
            self.create_normal_ssh_session(),
            self.create_normal_dns_queries(),
            self.create_ddos_attack(),
            self.create_port_scan(),
            self.create_ssh_brute_force(),
            self.create_dns_tunneling(),
            self.create_slowloris_attack()
        ]
        
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/{len(scenarios)}: {scenario['name']}")
            print(f"{'='*80}")
            print(f"Description: {scenario['description']}")
            print(f"Expected:    {scenario['expected']}")
            
            # Predict
            result = self.predict_scenario(scenario)
            
            # Display results
            print(f"\n📊 PREDICTION RESULTS:")
            print(f"  Predicted:   {result['predicted']}")
            print(f"  Confidence:  {result['confidence']:.4f}%")
            
            # Show probabilities
            print(f"\n  Probabilities:")
            for class_name, prob in result['probabilities'].items():
                bar = '█' * int(prob / 2)  # Scale for display
                print(f"    {class_name:12} {prob:6.4f}% {bar}")
            
            # Verdict
            if result['is_correct']:
                print(f"\n✅ CORRECT - Model accurately detected {scenario['expected']} traffic")
            else:
                print(f"\n❌ INCORRECT - Expected {scenario['expected']}, got {result['predicted']}")
            
            results.append({
                'scenario': scenario['name'],
                'expected': scenario['expected'],
                'predicted': result['predicted'],
                'confidence': result['confidence'],
                'correct': result['is_correct']
            })
        
        # Generate summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        # Calculate stats
        total_tests = len(results)
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = (correct_predictions / total_tests) * 100
        
        # Overall stats
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Tests:       {total_tests}")
        print(f"  Correct:           {correct_predictions}")
        print(f"  Incorrect:         {total_tests - correct_predictions}")
        print(f"  Accuracy:          {accuracy:.2f}%")
        
        # Average confidence
        avg_confidence = sum(r['confidence'] for r in results) / total_tests
        print(f"  Avg Confidence:    {avg_confidence:.4f}%")
        
        # Per-class breakdown
        print(f"\n📋 PER-CLASS RESULTS:")
        
        benign_results = [r for r in results if r['expected'] == 'Benign']
        malicious_results = [r for r in results if r['expected'] == 'Malicious']
        
        if benign_results:
            benign_correct = sum(1 for r in benign_results if r['correct'])
            benign_accuracy = (benign_correct / len(benign_results)) * 100
            print(f"\n  BENIGN TRAFFIC:")
            print(f"    Tests:      {len(benign_results)}")
            print(f"    Correct:    {benign_correct}")
            print(f"    Accuracy:   {benign_accuracy:.2f}%")
        
        if malicious_results:
            mal_correct = sum(1 for r in malicious_results if r['correct'])
            mal_accuracy = (mal_correct / len(malicious_results)) * 100
            print(f"\n  MALICIOUS TRAFFIC:")
            print(f"    Tests:      {len(malicious_results)}")
            print(f"    Correct:    {mal_correct}")
            print(f"    Accuracy:   {mal_accuracy:.2f}%")
        
        # Detailed results table
        print(f"\n📝 DETAILED RESULTS:")
        print(f"\n{'SCENARIO':<40} {'EXPECTED':<12} {'PREDICTED':<12} {'CONFIDENCE':<12} {'RESULT'}")
        print("="*90)
        
        for r in results:
            status = "✅ PASS" if r['correct'] else "❌ FAIL"
            print(f"{r['scenario']:<40} {r['expected']:<12} {r['predicted']:<12} "
                  f"{r['confidence']:>10.4f}%  {status}")
        
        # Final verdict
        print("\n" + "="*80)
        if accuracy == 100:
            print("🏆 PERFECT SCORE - All scenarios detected correctly!")
        elif accuracy >= 95:
            print("✅ EXCELLENT - Model performs exceptionally well on real-world scenarios")
        elif accuracy >= 85:
            print("✅ GOOD - Model performs well with minor issues")
        elif accuracy >= 70:
            print("⚠️  FAIR - Model needs improvement")
        else:
            print("❌ POOR - Model requires significant improvement")
        
        print("="*80)


# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SENTINET - REAL-WORLD SCENARIO TESTING")
    print("Testing trained model with aggressive, realistic attack patterns")
    print("="*80)
    
    try:
        # Initialize tester
        tester = RealWorldScenarioTester(
            model_path='models/tii_ssrc23_best_model_v1.pkl',
            scaler_path='processed_data/scaler.pkl',
            mapping_path='processed_data/attack_mapping.json'
        )
        
        # Run all tests
        results = tester.run_all_tests()
        
        print("\n✅ Testing complete!")
        print("\nResults saved in memory. Model performed:")
        
        accuracy = (sum(1 for r in results if r['correct']) / len(results)) * 100
        if accuracy == 100:
            print(f"  🏆 PERFECT: {accuracy:.2f}% accuracy on all real-world scenarios")
        else:
            print(f"  📊 Score: {accuracy:.2f}% accuracy")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠ Please ensure:")
        print("  1. Model file exists: models/tii_ssrc23_best_model_v1.pkl")
        print("  2. Scaler exists: processed_data/scaler.pkl")
        print("  3. Mapping exists: processed_data/attack_mapping.json")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
