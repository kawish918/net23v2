"""
Real-Time Network Intrusion Detection System (Phase 3)
======================================================
Captures live network packets, extracts features, and classifies traffic
using the trained LightGBM model.

Requirements:
- Run as Administrator (packet capture requires elevated privileges)
- Npcap installed (Windows packet capture driver)
- Both PCs on same network for attack testing

Author: SentiNet IDS
Date: October 2025
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
import warnings
warnings.filterwarnings('ignore')

class RealtimeIDS:
    def __init__(self, model_path, scaler_path, attack_mapping_path):
        """Initialize the Real-Time IDS"""
        print("="*80)
        print("SENTINET - REAL-TIME INTRUSION DETECTION SYSTEM")
        print("="*80)
        
        # Load model and preprocessing objects
        print("\n[+] Loading ML model and preprocessing objects...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load attack mapping
        import json
        with open(attack_mapping_path, 'r') as f:
            attack_map = json.load(f)
        # Reverse mapping: 0 -> 'Benign', 1 -> 'Malicious'
        self.label_map = {v: k for k, v in attack_map.items()}
        
        print(f"✓ Model loaded: {os.path.basename(model_path)}")
        print(f"✓ Scaler loaded: {os.path.basename(scaler_path)}")
        print(f"✓ Attack mapping: {self.label_map}")
        
        # Flow tracking (aggregate packets into flows)
        self.flows = defaultdict(lambda: {
            'packets': [],
            'start_time': None,
            'last_time': None,
            'packet_count': 0,
            'byte_count': 0,
            'syn_count': 0,
            'ack_count': 0,
            'psh_count': 0,
            'rst_count': 0,
            'fin_count': 0,
            'urg_count': 0
        })
        
        # Statistics
        self.stats = {
            'total_packets': 0,
            'total_flows': 0,
            'benign_count': 0,
            'malicious_count': 0,
            'start_time': time.time()
        }
        
        # Detection log
        self.detections = []
        self.log_file = 'logs/detections.csv'
        os.makedirs('logs', exist_ok=True)
        
        # Initialize log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('timestamp,src_ip,src_port,dst_ip,dst_port,protocol,prediction,confidence,flow_packets,flow_bytes\n')
        
        print(f"✓ Detection log: {self.log_file}")
        print("\n[+] IDS initialized successfully!")
        print("="*80)
    
    def get_flow_key(self, packet):
        """Generate unique flow identifier (5-tuple)"""
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            
            if TCP in packet:
                protocol = 'TCP'
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                protocol = 'UDP'
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            else:
                protocol = 'ICMP'
                src_port = 0
                dst_port = 0
            
            # Bidirectional flow key (sort to treat A->B and B->A as same flow)
            if (src_ip, src_port) < (dst_ip, dst_port):
                return (src_ip, src_port, dst_ip, dst_port, protocol)
            else:
                return (dst_ip, dst_port, src_ip, src_port, protocol)
        
        return None
    
    def extract_features(self, flow_key, flow_data):
        """
        Extract 79 features from flow data
        Note: This is a simplified version. Real implementation would need
        to match exact feature extraction from the TII-SSRC-23 dataset.
        """
        packets = flow_data['packets']
        if len(packets) == 0:
            return None
        
        # Basic flow information
        src_ip, src_port, dst_ip, dst_port, protocol = flow_key
        
        # Timing features
        duration = flow_data['last_time'] - flow_data['start_time'] if flow_data['last_time'] else 0
        
        # Packet features
        packet_lengths = [len(pkt) for pkt in packets]
        total_bytes = sum(packet_lengths)
        avg_packet_len = np.mean(packet_lengths) if packet_lengths else 0
        std_packet_len = np.std(packet_lengths) if len(packet_lengths) > 1 else 0
        max_packet_len = max(packet_lengths) if packet_lengths else 0
        min_packet_len = min(packet_lengths) if packet_lengths else 0
        
        # Rate features
        packet_rate = len(packets) / duration if duration > 0 else 0
        byte_rate = total_bytes / duration if duration > 0 else 0
        
        # TCP flags (if TCP)
        syn_ratio = flow_data['syn_count'] / len(packets) if len(packets) > 0 else 0
        ack_ratio = flow_data['ack_count'] / len(packets) if len(packets) > 0 else 0
        psh_ratio = flow_data['psh_count'] / len(packets) if len(packets) > 0 else 0
        rst_ratio = flow_data['rst_count'] / len(packets) if len(packets) > 0 else 0
        fin_ratio = flow_data['fin_count'] / len(packets) if len(packets) > 0 else 0
        urg_ratio = flow_data['urg_count'] / len(packets) if len(packets) > 0 else 0
        
        # Inter-arrival times
        iat_list = []
        for i in range(1, len(packets)):
            if hasattr(packets[i], 'time') and hasattr(packets[i-1], 'time'):
                iat = packets[i].time - packets[i-1].time
                iat_list.append(iat)
        
        avg_iat = np.mean(iat_list) if iat_list else 0
        std_iat = np.std(iat_list) if len(iat_list) > 1 else 0
        max_iat = max(iat_list) if iat_list else 0
        min_iat = min(iat_list) if iat_list else 0
        
        # Protocol encoding
        protocol_map = {'TCP': 6, 'UDP': 17, 'ICMP': 1}
        protocol_num = protocol_map.get(protocol, 0)
        
        # Build feature vector (79 features)
        # Note: This is a simplified feature set. Production system should match
        # exact features from training data (flow_duration, fwd_packet_length_max, etc.)
        features = [
            duration,                    # 0: Flow duration
            len(packets),                # 1: Total packets
            total_bytes,                 # 2: Total bytes
            avg_packet_len,              # 3: Average packet length
            std_packet_len,              # 4: Std dev packet length
            max_packet_len,              # 5: Max packet length
            min_packet_len,              # 6: Min packet length
            packet_rate,                 # 7: Packets per second
            byte_rate,                   # 8: Bytes per second
            syn_ratio,                   # 9: SYN flag ratio
            ack_ratio,                   # 10: ACK flag ratio
            psh_ratio,                   # 11: PSH flag ratio
            rst_ratio,                   # 12: RST flag ratio
            fin_ratio,                   # 13: FIN flag ratio
            urg_ratio,                   # 14: URG flag ratio
            avg_iat,                     # 15: Average inter-arrival time
            std_iat,                     # 16: Std dev IAT
            max_iat,                     # 17: Max IAT
            min_iat,                     # 18: Min IAT
            protocol_num,                # 19: Protocol number
            src_port,                    # 20: Source port
            dst_port,                    # 21: Destination port
        ]
        
        # Pad with zeros to reach 79 features (remaining features would be
        # forward/backward flow stats, header lengths, etc.)
        while len(features) < 79:
            features.append(0)
        
        return np.array(features[:79]).reshape(1, -1)
    
    def classify_flow(self, flow_key, flow_data):
        """Extract features and classify flow"""
        # Extract features
        features = self.extract_features(flow_key, flow_data)
        if features is None:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction] * 100
        
        # Get label
        label = self.label_map.get(prediction, 'Unknown')
        
        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def packet_handler(self, packet):
        """Process each captured packet"""
        self.stats['total_packets'] += 1
        
        # Get flow key
        flow_key = self.get_flow_key(packet)
        if flow_key is None:
            return
        
        # Update flow data
        flow = self.flows[flow_key]
        current_time = time.time()
        
        if flow['start_time'] is None:
            flow['start_time'] = current_time
        flow['last_time'] = current_time
        
        flow['packets'].append(packet)
        flow['packet_count'] += 1
        flow['byte_count'] += len(packet)
        
        # Extract TCP flags
        if TCP in packet:
            flags = packet[TCP].flags
            if flags & 0x02:  # SYN
                flow['syn_count'] += 1
            if flags & 0x10:  # ACK
                flow['ack_count'] += 1
            if flags & 0x08:  # PSH
                flow['psh_count'] += 1
            if flags & 0x04:  # RST
                flow['rst_count'] += 1
            if flags & 0x01:  # FIN
                flow['fin_count'] += 1
            if flags & 0x20:  # URG
                flow['urg_count'] += 1
        
        # Classify flow every 10 packets (or on FIN/RST)
        should_classify = False
        if flow['packet_count'] >= 10:
            should_classify = True
        elif TCP in packet and (packet[TCP].flags & 0x04 or packet[TCP].flags & 0x01):
            should_classify = True  # Classify on RST or FIN
        
        if should_classify:
            result = self.classify_flow(flow_key, flow)
            
            if result:
                src_ip, src_port, dst_ip, dst_port, protocol = flow_key
                
                # Update statistics
                if result['prediction'] == 'Benign':
                    self.stats['benign_count'] += 1
                else:
                    self.stats['malicious_count'] += 1
                
                # Display alert for malicious traffic
                if result['prediction'] == 'Malicious':
                    print(f"\n{'='*80}")
                    print(f"🚨 ALERT: {result['prediction'].upper()} TRAFFIC DETECTED!")
                    print(f"{'='*80}")
                    print(f"Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Source:       {src_ip}:{src_port}")
                    print(f"Destination:  {dst_ip}:{dst_port}")
                    print(f"Protocol:     {protocol}")
                    print(f"Confidence:   {result['confidence']:.4f}%")
                    print(f"Flow Stats:   {flow['packet_count']} packets, {flow['byte_count']} bytes")
                    print(f"Probabilities: Benign={result['probabilities'][0]*100:.2f}%, Malicious={result['probabilities'][1]*100:.2f}%")
                    print(f"{'='*80}\n")
                
                # Log detection
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = f"{timestamp},{src_ip},{src_port},{dst_ip},{dst_port},{protocol},{result['prediction']},{result['confidence']:.4f},{flow['packet_count']},{flow['byte_count']}\n"
                
                with open(self.log_file, 'a') as f:
                    f.write(log_entry)
                
                # Clear flow (start fresh)
                del self.flows[flow_key]
                self.stats['total_flows'] += 1
    
    def display_stats(self):
        """Display real-time statistics"""
        runtime = time.time() - self.stats['start_time']
        packets_per_sec = self.stats['total_packets'] / runtime if runtime > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"REAL-TIME STATISTICS")
        print(f"{'='*80}")
        print(f"Runtime:          {runtime:.1f} seconds")
        print(f"Total Packets:    {self.stats['total_packets']:,}")
        print(f"Total Flows:      {self.stats['total_flows']:,}")
        print(f"Active Flows:     {len(self.flows):,}")
        print(f"Packet Rate:      {packets_per_sec:.1f} packets/sec")
        print(f"Benign Traffic:   {self.stats['benign_count']:,}")
        print(f"Malicious Traffic: {self.stats['malicious_count']:,}")
        print(f"Detection Rate:   {(self.stats['malicious_count']/(self.stats['total_flows'] or 1))*100:.2f}%")
        print(f"{'='*80}\n")
    
    def start_capture(self, interface=None, packet_count=0):
        """Start packet capture"""
        print(f"\n[+] Starting packet capture...")
        print(f"    Interface: {interface or 'All interfaces'}")
        print(f"    Packet limit: {packet_count if packet_count > 0 else 'Unlimited'}")
        print(f"\n[!] Press Ctrl+C to stop capture\n")
        print("="*80)
        
        try:
            # Start statistics display thread
            import threading
            def stats_display():
                while True:
                    time.sleep(30)  # Display stats every 30 seconds
                    self.display_stats()
            
            stats_thread = threading.Thread(target=stats_display, daemon=True)
            stats_thread.start()
            
            # Start packet capture
            sniff(
                iface=interface,
                prn=self.packet_handler,
                store=False,
                count=packet_count
            )
        
        except KeyboardInterrupt:
            print("\n\n[!] Capture stopped by user")
            self.display_stats()
            print(f"\n[+] Detection log saved to: {self.log_file}")
            print("\n[+] IDS shutdown complete")
        
        except Exception as e:
            print(f"\n[!] Error during capture: {e}")
            print("\n[!] Make sure you're running as Administrator!")
            sys.exit(1)


def main():
    """Main entry point"""
    # Paths
    model_path = 'models/tii_ssrc23_best_model_v1.pkl'
    scaler_path = 'processed_data/scaler.pkl'
    attack_mapping_path = 'processed_data/attack_mapping.json'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}")
        print("[!] Please train the model first (run src/train_model.py)")
        sys.exit(1)
    
    if not os.path.exists(scaler_path):
        print(f"[!] Error: Scaler not found at {scaler_path}")
        print("[!] Please run preprocessing first (src/preprocessing.py)")
        sys.exit(1)
    
    # Initialize IDS
    ids = RealtimeIDS(model_path, scaler_path, attack_mapping_path)
    
    # Start capture
    # Note: Leave interface=None to capture on all interfaces
    # Or specify interface like interface='Wi-Fi' or interface='Ethernet'
    ids.start_capture(interface=None, packet_count=0)


if __name__ == '__main__':
    main()
