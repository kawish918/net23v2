"""
SentiNet - Real-Time IDS with PySide6 GUI
==========================================
Graphical interface for network intrusion detection system.
Displays live packet capture, detections, and statistics.

Requirements:
- Run as Administrator (packet capture requires elevated privileges)
- Npcap installed on Windows
- All dependencies from requirements.txt

Author: SentiNet IDS
Date: October 2025
"""

import os
import sys
import time
import joblib
import numpy as np
from datetime import datetime
from collections import defaultdict
from threading import Thread, Lock
import warnings
warnings.filterwarnings('ignore')

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QComboBox, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QColor

from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list


class IDSSignals(QObject):
    """Signals for thread-safe GUI updates"""
    packet_captured = Signal(str)
    detection_alert = Signal(dict)
    stats_updated = Signal(dict)


class NetworkIDS:
    """Core IDS functionality (non-GUI)"""
    
    def __init__(self, model_path, scaler_path, attack_mapping_path):
        """Initialize IDS engine"""
        # Load model and preprocessing
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load attack mapping
        import json
        with open(attack_mapping_path, 'r') as f:
            attack_map = json.load(f)
        self.label_map = {v: k for k, v in attack_map.items()}
        
        # Flow tracking
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
        
        # Thread safety
        self.lock = Lock()
        self.running = False
        
        # Signals for GUI updates
        self.signals = IDSSignals()
    
    def get_flow_key(self, packet):
        """Generate unique flow identifier"""
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
            
            if (src_ip, src_port) < (dst_ip, dst_port):
                return (src_ip, src_port, dst_ip, dst_port, protocol)
            else:
                return (dst_ip, dst_port, src_ip, src_port, protocol)
        
        return None
    
    def extract_features(self, flow_key, flow_data):
        """Extract 79 features from flow"""
        packets = flow_data['packets']
        if len(packets) == 0:
            return None
        
        src_ip, src_port, dst_ip, dst_port, protocol = flow_key
        
        # Timing
        duration = flow_data['last_time'] - flow_data['start_time'] if flow_data['last_time'] else 0
        
        # Packet stats
        packet_lengths = [len(pkt) for pkt in packets]
        total_bytes = sum(packet_lengths)
        avg_packet_len = np.mean(packet_lengths) if packet_lengths else 0
        std_packet_len = np.std(packet_lengths) if len(packet_lengths) > 1 else 0
        max_packet_len = max(packet_lengths) if packet_lengths else 0
        min_packet_len = min(packet_lengths) if packet_lengths else 0
        
        # Rates
        packet_rate = len(packets) / duration if duration > 0 else 0
        byte_rate = total_bytes / duration if duration > 0 else 0
        
        # TCP flags
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
        
        # Protocol
        protocol_map = {'TCP': 6, 'UDP': 17, 'ICMP': 1}
        protocol_num = protocol_map.get(protocol, 0)
        
        # Build feature vector (79 features)
        features = [
            duration, len(packets), total_bytes, avg_packet_len, std_packet_len,
            max_packet_len, min_packet_len, packet_rate, byte_rate,
            syn_ratio, ack_ratio, psh_ratio, rst_ratio, fin_ratio, urg_ratio,
            avg_iat, std_iat, max_iat, min_iat, protocol_num, src_port, dst_port
        ]
        
        # Pad to 79 features
        while len(features) < 79:
            features.append(0)
        
        return np.array(features[:79]).reshape(1, -1)
    
    def classify_flow(self, flow_key, flow_data):
        """Classify flow as benign or malicious"""
        features = self.extract_features(flow_key, flow_data)
        if features is None:
            return None
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction] * 100
        
        label = self.label_map.get(prediction, 'Unknown')
        
        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': probabilities,
            'flow_key': flow_key,
            'flow_data': flow_data
        }
    
    def packet_handler(self, packet):
        """Process captured packet"""
        with self.lock:
            self.stats['total_packets'] += 1
        
        flow_key = self.get_flow_key(packet)
        if flow_key is None:
            return
        
        # Update flow
        flow = self.flows[flow_key]
        current_time = time.time()
        
        if flow['start_time'] is None:
            flow['start_time'] = current_time
        flow['last_time'] = current_time
        flow['packets'].append(packet)
        flow['packet_count'] += 1
        flow['byte_count'] += len(packet)
        
        # TCP flags
        if TCP in packet:
            flags = packet[TCP].flags
            if flags & 0x02: flow['syn_count'] += 1
            if flags & 0x10: flow['ack_count'] += 1
            if flags & 0x08: flow['psh_count'] += 1
            if flags & 0x04: flow['rst_count'] += 1
            if flags & 0x01: flow['fin_count'] += 1
            if flags & 0x20: flow['urg_count'] += 1
        
        # Emit packet info
        src_ip, src_port, dst_ip, dst_port, protocol = flow_key
        packet_info = f"{src_ip}:{src_port} → {dst_ip}:{dst_port} [{protocol}]"
        self.signals.packet_captured.emit(packet_info)
        
        # Classify every 10 packets or on connection close
        should_classify = flow['packet_count'] >= 10
        if TCP in packet and (packet[TCP].flags & 0x04 or packet[TCP].flags & 0x01):
            should_classify = True
        
        if should_classify:
            result = self.classify_flow(flow_key, flow)
            if result:
                with self.lock:
                    if result['prediction'] == 'Benign':
                        self.stats['benign_count'] += 1
                    else:
                        self.stats['malicious_count'] += 1
                    self.stats['total_flows'] += 1
                
                # Emit detection
                self.signals.detection_alert.emit(result)
                
                # Clear flow
                del self.flows[flow_key]
        
        # Emit stats update
        with self.lock:
            stats_copy = self.stats.copy()
        self.signals.stats_updated.emit(stats_copy)
    
    def start_capture(self, interface=None):
        """Start packet capture in background thread"""
        self.running = True
        
        def capture_loop():
            try:
                sniff(
                    iface=interface,
                    prn=self.packet_handler,
                    store=False,
                    stop_filter=lambda x: not self.running
                )
            except Exception as e:
                print(f"Capture error: {e}")
        
        Thread(target=capture_loop, daemon=True).start()
    
    def stop_capture(self):
        """Stop packet capture"""
        self.running = False


class IDSMainWindow(QMainWindow):
    """Main GUI Window"""
    
    def __init__(self):
        super().__init__()
        
        self.ids = None
        self.init_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.start(1000)  # Update every second
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("SentiNet - Real-Time Intrusion Detection System")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("🛡️ SentiNet IDS - Live Network Monitor")
        header_font = QFont("Arial", 18, QFont.Bold)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("background-color: #2c3e50; color: white; padding: 15px; border-radius: 5px;")
        main_layout.addWidget(header)
        
        # Control panel
        control_group = self.create_control_panel()
        main_layout.addWidget(control_group)
        
        # Statistics panel
        stats_layout = QHBoxLayout()
        self.stats_labels = self.create_stats_panel()
        stats_layout.addLayout(self.stats_labels)
        main_layout.addLayout(stats_layout)
        
        # Content area (2 columns)
        content_layout = QHBoxLayout()
        
        # Left: Packet stream
        left_layout = QVBoxLayout()
        packet_label = QLabel("📡 Live Packet Stream")
        packet_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(packet_label)
        
        self.packet_stream = QTextEdit()
        self.packet_stream.setReadOnly(True)
        self.packet_stream.setMaximumHeight(250)
        self.packet_stream.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        left_layout.addWidget(self.packet_stream)
        
        # Detections table
        detections_label = QLabel("🚨 Detection Alerts")
        detections_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(detections_label)
        
        self.detections_table = QTableWidget()
        self.detections_table.setColumnCount(6)
        self.detections_table.setHorizontalHeaderLabels([
            "Time", "Source", "Destination", "Protocol", "Classification", "Confidence"
        ])
        self.detections_table.setStyleSheet("background-color: white;")
        left_layout.addWidget(self.detections_table)
        
        content_layout.addLayout(left_layout, 60)
        
        # Right: Activity log
        right_layout = QVBoxLayout()
        log_label = QLabel("📋 Activity Log")
        log_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(log_label)
        
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setStyleSheet("background-color: #f8f9fa; font-family: Consolas;")
        right_layout.addWidget(self.activity_log)
        
        content_layout.addLayout(right_layout, 40)
        
        main_layout.addLayout(content_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready - Click 'Start Capture' to begin monitoring")
    
    def create_control_panel(self):
        """Create control buttons"""
        group = QGroupBox("Control Panel")
        layout = QHBoxLayout()
        
        # Interface selection
        layout.addWidget(QLabel("Network Interface:"))
        self.interface_combo = QComboBox()
        self.interface_combo.addItem("All Interfaces", None)
        try:
            for iface in get_if_list():
                self.interface_combo.addItem(iface, iface)
        except:
            pass
        layout.addWidget(self.interface_combo)
        
        layout.addStretch()
        
        # Start button
        self.start_btn = QPushButton("▶ Start Capture")
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self.start_capture)
        layout.addWidget(self.start_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop Capture")
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 8px;")
        self.stop_btn.clicked.connect(self.stop_capture)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear Logs")
        clear_btn.clicked.connect(self.clear_logs)
        layout.addWidget(clear_btn)
        
        group.setLayout(layout)
        return group
    
    def create_stats_panel(self):
        """Create statistics display"""
        layout = QHBoxLayout()
        
        self.stat_widgets = {}
        stats = [
            ("📦 Packets", "packets", "#3498db"),
            ("🔄 Flows", "flows", "#9b59b6"),
            ("✅ Benign", "benign", "#27ae60"),
            ("🚨 Malicious", "malicious", "#e74c3c"),
            ("📊 Rate", "rate", "#f39c12")
        ]
        
        for label, key, color in stats:
            widget = QGroupBox(label)
            widget_layout = QVBoxLayout()
            
            value_label = QLabel("0")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
            widget_layout.addWidget(value_label)
            
            widget.setLayout(widget_layout)
            layout.addWidget(widget)
            
            self.stat_widgets[key] = value_label
        
        return layout
    
    def start_capture(self):
        """Start IDS capture"""
        try:
            # Check if model exists
            model_path = 'models/tii_ssrc23_best_model_v1.pkl'
            scaler_path = 'processed_data/scaler.pkl'
            mapping_path = 'processed_data/attack_mapping.json'
            
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "Error", f"Model not found: {model_path}")
                return
            
            # Initialize IDS
            self.log_activity("Initializing IDS engine...")
            self.ids = NetworkIDS(model_path, scaler_path, mapping_path)
            
            # Connect signals
            self.ids.signals.packet_captured.connect(self.on_packet_captured)
            self.ids.signals.detection_alert.connect(self.on_detection_alert)
            self.ids.signals.stats_updated.connect(self.on_stats_updated)
            
            # Start capture
            interface = self.interface_combo.currentData()
            self.ids.start_capture(interface)
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("✅ Capturing - Monitoring network traffic in real-time")
            self.log_activity("🚀 IDS started successfully!")
            self.log_activity(f"📡 Monitoring interface: {interface or 'All'}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start capture:\n{str(e)}\n\nMake sure you're running as Administrator!")
            self.log_activity(f"❌ Error: {str(e)}")
    
    def stop_capture(self):
        """Stop IDS capture"""
        if self.ids:
            self.ids.stop_capture()
            self.log_activity("⏹ IDS stopped")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Stopped - Ready to restart")
    
    def clear_logs(self):
        """Clear all logs and detections"""
        self.packet_stream.clear()
        self.activity_log.clear()
        self.detections_table.setRowCount(0)
        self.log_activity("🗑️ Logs cleared")
    
    def on_packet_captured(self, packet_info):
        """Handle packet capture signal"""
        self.packet_stream.append(packet_info)
        
        # Keep only last 100 lines
        doc = self.packet_stream.document()
        if doc.blockCount() > 100:
            cursor = self.packet_stream.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
    
    def on_detection_alert(self, result):
        """Handle detection alert signal"""
        src_ip, src_port, dst_ip, dst_port, protocol = result['flow_key']
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Add to table
        row = self.detections_table.rowCount()
        self.detections_table.insertRow(row)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        items = [
            timestamp,
            f"{src_ip}:{src_port}",
            f"{dst_ip}:{dst_port}",
            protocol,
            prediction,
            f"{confidence:.2f}%"
        ]
        
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            
            # Color code malicious traffic
            if prediction == "Malicious":
                item.setBackground(QColor(255, 200, 200))
            
            self.detections_table.setItem(row, col, item)
        
        # Scroll to bottom
        self.detections_table.scrollToBottom()
        
        # Log activity
        if prediction == "Malicious":
            self.log_activity(f"🚨 MALICIOUS: {src_ip}:{src_port} → {dst_ip}:{dst_port} [{protocol}] (Confidence: {confidence:.2f}%)")
    
    def on_stats_updated(self, stats):
        """Handle stats update signal"""
        # Stats will be updated by timer
        pass
    
    def update_displays(self):
        """Update statistics display"""
        if self.ids:
            stats = self.ids.stats
            runtime = time.time() - stats['start_time']
            rate = stats['total_packets'] / runtime if runtime > 0 else 0
            
            self.stat_widgets['packets'].setText(f"{stats['total_packets']:,}")
            self.stat_widgets['flows'].setText(f"{stats['total_flows']:,}")
            self.stat_widgets['benign'].setText(f"{stats['benign_count']:,}")
            self.stat_widgets['malicious'].setText(f"{stats['malicious_count']:,}")
            self.stat_widgets['rate'].setText(f"{rate:.1f}/s")
    
    def log_activity(self, message):
        """Log activity message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.append(f"[{timestamp}] {message}")


def main():
    """Main entry point"""
    # Check if running as admin (Windows)
    import ctypes
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False
    
    if not is_admin:
        print("⚠️  WARNING: Not running as Administrator!")
        print("   Packet capture may fail. Please run as Administrator.")
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = IDSMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
