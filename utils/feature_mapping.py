"""
SentiNet - Feature Mapping Utilities
Maps live packet data to model-compatible features

This module defines how to extract features from live network packets
that match the training data structure.
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime


class FeatureExtractor:
    """
    Extracts and maps network packet features for real-time inference
    """
    
    def __init__(self, feature_mapping_path='processed_data/feature_mapping.json'):
        """
        Initialize feature extractor with mapping configuration
        
        Args:
            feature_mapping_path: Path to feature mapping JSON file
        """
        with open(feature_mapping_path, 'r') as f:
            self.mapping = json.load(f)
        
        self.feature_names = self.mapping['all_features']
        self.num_features = len(self.feature_names)
        
        # Initialize flow tracking for stateful features
        self.flow_tracker = FlowTracker()
        
        print(f"✓ Feature extractor initialized with {self.num_features} features")
    
    def extract_from_packet(self, packet_data):
        """
        Extract features from a single packet
        
        Args:
            packet_data: Dictionary containing packet information
                Expected keys: src_ip, dst_ip, src_port, dst_port, protocol,
                              packet_length, tcp_flags, timestamp, etc.
        
        Returns:
            numpy array of extracted features (aligned with training data)
        """
        features = np.zeros(self.num_features)
        
        # Map packet data to feature vector
        # This is a template - adjust based on actual TII-SSRC-23 features
        
        feature_dict = {}
        
        # Basic packet features
        if 'packet_length' in packet_data or 'pkt_len' in packet_data:
            feature_dict['pkt_len'] = packet_data.get('packet_length', 
                                                     packet_data.get('pkt_len', 0))
        
        # Protocol information
        if 'protocol' in packet_data:
            protocol_map = {'TCP': 6, 'UDP': 17, 'ICMP': 1}
            feature_dict['protocol'] = protocol_map.get(packet_data['protocol'], 0)
        
        # Port information
        feature_dict['src_port'] = packet_data.get('src_port', 0)
        feature_dict['dst_port'] = packet_data.get('dst_port', 0)
        
        # TCP flags (if TCP packet)
        if 'tcp_flags' in packet_data:
            flags = packet_data['tcp_flags']
            feature_dict['syn_flag'] = 1 if flags.get('SYN', False) else 0
            feature_dict['ack_flag'] = 1 if flags.get('ACK', False) else 0
            feature_dict['psh_flag'] = 1 if flags.get('PSH', False) else 0
            feature_dict['rst_flag'] = 1 if flags.get('RST', False) else 0
            feature_dict['fin_flag'] = 1 if flags.get('FIN', False) else 0
        
        # Time to live
        feature_dict['ttl'] = packet_data.get('ttl', 64)
        
        # Flow-based features (requires flow tracking)
        flow_key = self._get_flow_key(packet_data)
        flow_features = self.flow_tracker.update_flow(flow_key, packet_data)
        feature_dict.update(flow_features)
        
        # Map extracted features to feature vector
        for i, feature_name in enumerate(self.feature_names):
            # Try exact match
            if feature_name in feature_dict:
                features[i] = feature_dict[feature_name]
            # Try lowercase match
            elif feature_name.lower() in feature_dict:
                features[i] = feature_dict[feature_name.lower()]
            # Try partial match
            else:
                for key in feature_dict:
                    if key in feature_name.lower() or feature_name.lower() in key:
                        features[i] = feature_dict[key]
                        break
        
        return features
    
    def _get_flow_key(self, packet_data):
        """
        Generate unique flow identifier
        
        Args:
            packet_data: Packet information dictionary
        
        Returns:
            Tuple representing bidirectional flow
        """
        src_ip = packet_data.get('src_ip', '0.0.0.0')
        dst_ip = packet_data.get('dst_ip', '0.0.0.0')
        src_port = packet_data.get('src_port', 0)
        dst_port = packet_data.get('dst_port', 0)
        protocol = packet_data.get('protocol', 'TCP')
        
        # Create bidirectional flow key (sorted to match both directions)
        endpoints = sorted([(src_ip, src_port), (dst_ip, dst_port)])
        return (endpoints[0], endpoints[1], protocol)
    
    def extract_batch(self, packet_list):
        """
        Extract features from multiple packets
        
        Args:
            packet_list: List of packet dictionaries
        
        Returns:
            numpy array of shape (n_packets, n_features)
        """
        features_batch = np.zeros((len(packet_list), self.num_features))
        
        for i, packet in enumerate(packet_list):
            features_batch[i] = self.extract_from_packet(packet)
        
        return features_batch
    
    def reset_flow_tracker(self):
        """Reset flow tracking state (useful for testing or clearing old flows)"""
        self.flow_tracker = FlowTracker()
        print("✓ Flow tracker reset")


class FlowTracker:
    """
    Tracks network flows to compute stateful features
    (duration, packet counts, byte counts, inter-arrival times, etc.)
    """
    
    def __init__(self, timeout=120):
        """
        Initialize flow tracker
        
        Args:
            timeout: Flow timeout in seconds (flows inactive for this duration are removed)
        """
        self.flows = {}
        self.timeout = timeout
        self.last_cleanup = datetime.now()
    
    def update_flow(self, flow_key, packet_data):
        """
        Update flow statistics with new packet
        
        Args:
            flow_key: Unique flow identifier
            packet_data: Packet information
        
        Returns:
            Dictionary of flow-based features
        """
        current_time = datetime.now()
        
        # Initialize flow if new
        if flow_key not in self.flows:
            self.flows[flow_key] = {
                'start_time': current_time,
                'last_time': current_time,
                'fwd_pkts': 0,
                'bwd_pkts': 0,
                'fwd_bytes': 0,
                'bwd_bytes': 0,
                'packet_lengths': [],
                'inter_arrival_times': [],
                'syn_count': 0,
                'ack_count': 0,
                'psh_count': 0,
                'rst_count': 0,
                'fin_count': 0
            }
        
        flow = self.flows[flow_key]
        
        # Determine packet direction (forward/backward)
        is_forward = self._is_forward_packet(flow_key, packet_data)
        
        # Update packet and byte counts
        pkt_len = packet_data.get('packet_length', packet_data.get('pkt_len', 0))
        
        if is_forward:
            flow['fwd_pkts'] += 1
            flow['fwd_bytes'] += pkt_len
        else:
            flow['bwd_pkts'] += 1
            flow['bwd_bytes'] += pkt_len
        
        flow['packet_lengths'].append(pkt_len)
        
        # Calculate inter-arrival time
        if flow['last_time'] != flow['start_time']:
            iat = (current_time - flow['last_time']).total_seconds()
            flow['inter_arrival_times'].append(iat)
        
        flow['last_time'] = current_time
        
        # Update flag counts
        if 'tcp_flags' in packet_data:
            flags = packet_data['tcp_flags']
            if flags.get('SYN', False): flow['syn_count'] += 1
            if flags.get('ACK', False): flow['ack_count'] += 1
            if flags.get('PSH', False): flow['psh_count'] += 1
            if flags.get('RST', False): flow['rst_count'] += 1
            if flags.get('FIN', False): flow['fin_count'] += 1
        
        # Compute flow features
        features = self._compute_flow_features(flow)
        
        # Periodic cleanup of old flows
        if (current_time - self.last_cleanup).total_seconds() > 60:
            self._cleanup_old_flows(current_time)
            self.last_cleanup = current_time
        
        return features
    
    def _is_forward_packet(self, flow_key, packet_data):
        """
        Determine if packet is in forward direction of flow
        
        Args:
            flow_key: Flow identifier
            packet_data: Packet information
        
        Returns:
            Boolean indicating forward direction
        """
        src = (packet_data.get('src_ip', '0.0.0.0'), 
               packet_data.get('src_port', 0))
        
        # First endpoint in flow_key is considered forward direction
        return src == flow_key[0]
    
    def _compute_flow_features(self, flow):
        """
        Compute statistical features from flow data
        
        Args:
            flow: Flow statistics dictionary
        
        Returns:
            Dictionary of computed features
        """
        features = {}
        
        # Duration
        duration = (flow['last_time'] - flow['start_time']).total_seconds()
        features['duration'] = duration
        
        # Packet counts
        features['fwd_pkts'] = flow['fwd_pkts']
        features['bwd_pkts'] = flow['bwd_pkts']
        features['flow_pkts'] = flow['fwd_pkts'] + flow['bwd_pkts']
        
        # Byte counts
        features['fwd_bytes'] = flow['fwd_bytes']
        features['bwd_bytes'] = flow['bwd_bytes']
        features['flow_bytes'] = flow['fwd_bytes'] + flow['bwd_bytes']
        
        # Packet length statistics
        if flow['packet_lengths']:
            pkt_lens = flow['packet_lengths']
            features['pkt_len_mean'] = np.mean(pkt_lens)
            features['pkt_len_std'] = np.std(pkt_lens)
            features['pkt_len_max'] = np.max(pkt_lens)
            features['pkt_len_min'] = np.min(pkt_lens)
        else:
            features['pkt_len_mean'] = 0
            features['pkt_len_std'] = 0
            features['pkt_len_max'] = 0
            features['pkt_len_min'] = 0
        
        # Inter-arrival time statistics
        if flow['inter_arrival_times']:
            iats = flow['inter_arrival_times']
            features['iat_mean'] = np.mean(iats)
            features['iat_std'] = np.std(iats)
            features['iat_max'] = np.max(iats)
            features['iat_min'] = np.min(iats)
        else:
            features['iat_mean'] = 0
            features['iat_std'] = 0
            features['iat_max'] = 0
            features['iat_min'] = 0
        
        # Rate features (packets/bytes per second)
        if duration > 0:
            features['fwd_pkt_rate'] = flow['fwd_pkts'] / duration
            features['bwd_pkt_rate'] = flow['bwd_pkts'] / duration
            features['flow_byte_rate'] = features['flow_bytes'] / duration
        else:
            features['fwd_pkt_rate'] = 0
            features['bwd_pkt_rate'] = 0
            features['flow_byte_rate'] = 0
        
        # Flag counts
        features['syn_count'] = flow['syn_count']
        features['ack_count'] = flow['ack_count']
        features['psh_count'] = flow['psh_count']
        features['rst_count'] = flow['rst_count']
        features['fin_count'] = flow['fin_count']
        
        # Ratios
        total_pkts = features['flow_pkts']
        if total_pkts > 0:
            features['fwd_pkt_ratio'] = flow['fwd_pkts'] / total_pkts
            features['bwd_pkt_ratio'] = flow['bwd_pkts'] / total_pkts
        else:
            features['fwd_pkt_ratio'] = 0
            features['bwd_pkt_ratio'] = 0
        
        return features
    
    def _cleanup_old_flows(self, current_time):
        """
        Remove flows that have been inactive for longer than timeout
        
        Args:
            current_time: Current timestamp
        """
        flows_to_remove = []
        
        for flow_key, flow_data in self.flows.items():
            inactive_time = (current_time - flow_data['last_time']).total_seconds()
            if inactive_time > self.timeout:
                flows_to_remove.append(flow_key)
        
        for flow_key in flows_to_remove:
            del self.flows[flow_key]
        
        if flows_to_remove:
            print(f"  Cleaned up {len(flows_to_remove)} inactive flows")
    
    def get_active_flows(self):
        """Return number of currently tracked flows"""
        return len(self.flows)
    
    def clear_all(self):
        """Clear all tracked flows"""
        self.flows = {}


# Feature name standardization mappings
FEATURE_ALIASES = {
    'packet_length': ['pkt_len', 'packet_len', 'length', 'len'],
    'src_port': ['source_port', 'sport'],
    'dst_port': ['destination_port', 'dport', 'dest_port'],
    'src_ip': ['source_ip', 'sip'],
    'dst_ip': ['destination_ip', 'dip', 'dest_ip'],
    'protocol': ['proto', 'ip_proto'],
    'ttl': ['time_to_live', 'ip_ttl'],
}


def normalize_feature_name(name):
    """
    Normalize feature name to standard format
    
    Args:
        name: Feature name string
    
    Returns:
        Standardized feature name
    """
    name_lower = name.lower().strip()
    
    # Check against aliases
    for standard_name, aliases in FEATURE_ALIASES.items():
        if name_lower in aliases or name_lower == standard_name:
            return standard_name
    
    return name_lower


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("FEATURE EXTRACTION MODULE TEST")
    print("="*80)
    
    # Initialize extractor
    try:
        extractor = FeatureExtractor('processed_data/feature_mapping.json')
        print("\n✓ Feature extractor loaded successfully")
    except FileNotFoundError:
        print("\n⚠ Feature mapping not found. Run preprocessing first.")
        print("  Creating mock extractor for testing...")
        
        # Create mock mapping for testing
        import os
        os.makedirs('processed_data', exist_ok=True)
        mock_mapping = {
            'all_features': [
                'pkt_len', 'protocol', 'src_port', 'dst_port', 'duration',
                'fwd_pkts', 'bwd_pkts', 'flow_bytes', 'iat_mean', 'syn_flag'
            ],
            'feature_categories': {},
            'label_column': 'label'
        }
        with open('processed_data/feature_mapping.json', 'w') as f:
            json.dump(mock_mapping, f, indent=2)
        
        extractor = FeatureExtractor('processed_data/feature_mapping.json')
    
    # Test with sample packet
    print("\n" + "-"*80)
    print("Testing feature extraction from sample packet...")
    print("-"*80)
    
    sample_packet = {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.50',
        'src_port': 54321,
        'dst_port': 80,
        'protocol': 'TCP',
        'packet_length': 1460,
        'ttl': 64,
        'tcp_flags': {'SYN': True, 'ACK': False, 'PSH': False, 'RST': False, 'FIN': False},
        'timestamp': datetime.now()
    }
    
    features = extractor.extract_from_packet(sample_packet)
    
    print(f"\nExtracted feature vector shape: {features.shape}")
    print(f"Non-zero features: {np.count_nonzero(features)}/{len(features)}")
    print(f"\nSample feature values:")
    for i in range(min(10, len(features))):
        print(f"  Feature {i}: {features[i]:.4f}")
    
    # Test batch extraction
    print("\n" + "-"*80)
    print("Testing batch extraction...")
    print("-"*80)
    
    packet_batch = [sample_packet for _ in range(5)]
    batch_features = extractor.extract_batch(packet_batch)
    
    print(f"\nBatch feature matrix shape: {batch_features.shape}")
    print("✓ Feature extraction test complete")
    
    # Flow tracker test
    print("\n" + "-"*80)
    print("Testing flow tracker...")
    print("-"*80)
    
    print(f"\nActive flows: {extractor.flow_tracker.get_active_flows()}")
    extractor.reset_flow_tracker()
    print("✓ Flow tracker test complete")