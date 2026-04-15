#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Industrial-grade FPV and FM signal detection script
Function: Comprehensive signal scanning, detection, demodulation, and visualization
Device: USRP B210 / B200mini
Tech stack: Python 3.8-3.11 + uhd + numpy + scipy + matplotlib
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import threading
import queue
import json
import os
import logging
import uhd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fpv_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('industrial_fpv_scanner')

# Try to import pyaudio for audio output
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("pyaudio not installed, audio output will be unavailable")

class DeviceManager:
    """Device management class for USRP"""
    
    def __init__(self):
        """Initialize USRP device"""
        self.usrp = None
        self.sample_rate = None
        self.gain = None
        self.antenna = None
        self.is_initialized = False
    
    def initialize(self, sample_rate=2e6, gain=30, antenna="RX2"):
        """Initialize USRP device with error handling"""
        try:
            logger.info("Initializing USRP device...")
            
            # Create USRP device
            self.usrp = uhd.usrp.MultiUSRP()
            
            # Store parameters
            self.sample_rate = sample_rate
            self.gain = gain
            self.antenna = antenna
            
            # Set sample rate
            logger.info(f"Setting sample rate: {self.sample_rate / 1e6} MHz")
            self.usrp.set_rx_rate(self.sample_rate, 0)
            actual_rate = self.usrp.get_rx_rate(0)
            logger.info(f"Actual sample rate: {actual_rate / 1e6} MHz")
            
            # Set gain
            logger.info(f"Setting gain: {self.gain} dB")
            self.usrp.set_rx_gain(self.gain, 0)
            actual_gain = self.usrp.get_rx_gain(0)
            logger.info(f"Actual gain: {actual_gain} dB")
            
            # Set antenna
            logger.info(f"Setting antenna: {self.antenna}")
            self.usrp.set_rx_antenna(self.antenna, 0)
            actual_antenna = self.usrp.get_rx_antenna(0)
            logger.info(f"Actual antenna: {actual_antenna}")
            
            # Print device info
            logger.info(f"USRP initialization successful: {self.usrp.get_pp_string()}")
            
            # Try to get frequency range
            try:
                freq_range = self.usrp.get_rx_freq_range(0)
                logger.info(f"Device frequency range: {freq_range.start / 1e6:.2f} - {freq_range.stop / 1e6:.2f} MHz")
            except Exception as e:
                logger.warning(f"Failed to get frequency range: {e}")
            
            # Try to get sensor info
            try:
                sensor_names = self.usrp.get_rx_sensor_names(0)
                logger.info(f"Device sensors: {sensor_names}")
                for sensor in sensor_names:
                    try:
                        value = self.usrp.get_rx_sensor(sensor, 0)
                        logger.info(f"  {sensor}: {value}")
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Failed to get sensor info: {e}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"USRP initialization failed: {e}")
            self.is_initialized = False
            return False
    
    def set_frequency(self, freq):
        """Set frequency, handle different API versions"""
        if not self.is_initialized or not self.usrp:
            logger.error("Device not initialized")
            return False
        
        try:
            logger.debug(f"Setting frequency: {freq / 1e6:.2f} MHz")
            
            # Try different API paths
            try:
                # Try using uhd.libpyuhd.types.tune_request
                if hasattr(uhd, 'libpyuhd') and hasattr(uhd.libpyuhd, 'types') and hasattr(uhd.libpyuhd.types, 'tune_request'):
                    tune_req = uhd.libpyuhd.types.tune_request(freq)
                    self.usrp.set_rx_freq(tune_req, 0)
                    logger.debug("  Using uhd.libpyuhd.types.tune_request")
                # Try using uhd.types.tune_request
                elif hasattr(uhd, 'types') and hasattr(uhd.types, 'tune_request'):
                    tune_req = uhd.types.tune_request(freq)
                    self.usrp.set_rx_freq(tune_req, 0)
                    logger.debug("  Using uhd.types.tune_request")
                else:
                    # Try setting frequency directly
                    self.usrp.set_rx_freq(freq, 0)
                    logger.debug("  Direct frequency setting")
            except Exception as e:
                logger.error(f"Failed to set frequency: {e}")
                return False
            
            # Verify frequency setting
            actual_freq = self.usrp.get_rx_freq(0)
            logger.debug(f"  Actual frequency: {actual_freq / 1e6:.2f} MHz")
            
            return True
        except Exception as e:
            logger.error(f"Failed to set frequency: {e}")
            return False
    
    def get_device_info(self):
        """Get device information"""
        if not self.is_initialized or not self.usrp:
            return {}
        
        try:
            info = {
                'pp_string': self.usrp.get_pp_string(),
                'sample_rate': self.usrp.get_rx_rate(0),
                'gain': self.usrp.get_rx_gain(0),
                'antenna': self.usrp.get_rx_antenna(0)
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}
    
    def close(self):
        """Close device"""
        try:
            if self.usrp:
                logger.info("Closing USRP device")
                # USRP objects are automatically cleaned up in Python
                self.usrp = None
                self.is_initialized = False
        except Exception as e:
            logger.error(f"Failed to close device: {e}")

class DataAcquisition:
    """Data acquisition class"""
    
    def __init__(self, device_manager):
        """Initialize data acquisition"""
        self.device_manager = device_manager
        self.rx_streamer = None
    
    def receive_samples(self, num_samples=8192, timeout=1.0):
        """Receive samples with error handling"""
        if not self.device_manager.is_initialized:
            logger.error("Device not initialized")
            return np.array([], dtype=np.complex64)
        
        try:
            # Construct stream_args
            stream_args = None
            
            # Priority use official recommended method
            try:
                if hasattr(uhd.usrp, 'StreamArgs'):
                    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
                    stream_args.channels = [0]
                    logger.debug("Using uhd.usrp.StreamArgs")
                else:
                    # Try dictionary format
                    stream_args = {
                        "cpu_format": "fc32",
                        "otw_format": "sc16",
                        "channels": [0]
                    }
                    logger.debug("Using dictionary format for stream_args")
            except Exception as e:
                logger.error(f"Failed to construct stream_args: {e}")
                return np.array([], dtype=np.complex64)
            
            # Create receive stream
            try:
                self.rx_streamer = self.device_manager.usrp.get_rx_stream(stream_args)
                logger.debug("Receive stream created successfully")
            except Exception as e:
                logger.error(f"Failed to create receive stream: {e}")
                return np.array([], dtype=np.complex64)
            
            # Allocate buffer
            buffer = np.zeros(num_samples, dtype=np.complex64)
            
            # Create metadata
            metadata = None
            try:
                # Try official recommended method
                from uhd import types
                if hasattr(types, 'RXMetadata'):
                    metadata = types.RXMetadata()
                    logger.debug("Using uhd.types.RXMetadata")
                else:
                    # Try other methods
                    from uhd.libpyuhd import types
                    if hasattr(types, 'rx_metadata_t'):
                        metadata = types.rx_metadata_t()
                        logger.debug("Using uhd.libpyuhd.types.rx_metadata_t")
                    elif hasattr(types, 'rx_metadata'):
                        metadata = types.rx_metadata()
                        logger.debug("Using uhd.libpyuhd.types.rx_metadata")
                    else:
                        logger.error("Failed to create metadata")
                        return np.array([], dtype=np.complex64)
            except Exception as e:
                logger.error(f"Failed to create metadata: {e}")
                return np.array([], dtype=np.complex64)
            
            # Send stream start command
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # Try using start mode
                    if hasattr(types.StreamMode, 'start'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start)
                        stream_cmd.stream_now = True
                        stream_cmd.num_samps = num_samples
                        self.rx_streamer.issue_stream_cmd(stream_cmd)
                        logger.debug("Stream start command sent (start mode)")
                    # Fallback to start_cont mode
                    elif hasattr(types.StreamMode, 'start_cont'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start_cont)
                        stream_cmd.stream_now = True
                        self.rx_streamer.issue_stream_cmd(stream_cmd)
                        logger.debug("Stream start command sent (start_cont mode)")
                    else:
                        logger.error("No suitable StreamMode found")
                        return np.array([], dtype=np.complex64)
                else:
                    # Try libpyuhd path
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # Try to find start mode
                        start_mode = None
                        # First try continuous mode
                        mode_options = ['STREAM_MODE_START_CONT', 'stream_mode_start_cont', 'START_CONT', 'STREAM_MODE_START_CONTINUOUS', 'start_continuous']
                        for mode_name in mode_options:
                            if hasattr(types.stream_cmd_t, mode_name):
                                start_mode = getattr(types.stream_cmd_t, mode_name)
                                break
                        
                        # If continuous mode not found, try one-shot mode
                        if start_mode is None:
                            mode_options = ['STREAM_MODE_START', 'stream_mode_start', 'START', 'STREAM_MODE_ONESHOT', 'oneshot']
                            for mode_name in mode_options:
                                if hasattr(types.stream_cmd_t, mode_name):
                                    start_mode = getattr(types.stream_cmd_t, mode_name)
                                    break
                        
                        if start_mode:
                            stream_cmd = types.stream_cmd_t(start_mode)
                            stream_cmd.stream_now = True
                            if hasattr(stream_cmd, 'num_samps'):
                                stream_cmd.num_samps = num_samples
                            self.rx_streamer.issue_stream_cmd(stream_cmd)
                            logger.debug("Stream start command sent (libpyuhd)")
                        else:
                            # Try using digital value directly
                            stream_cmd = types.stream_cmd_t(1)  # 1 is usually START_CONTINUOUS
                            stream_cmd.stream_now = True
                            self.rx_streamer.issue_stream_cmd(stream_cmd)
                            logger.debug("Stream start command sent (digital value 1)")
                    else:
                        logger.error("No stream_cmd_t found")
                        return np.array([], dtype=np.complex64)
            except Exception as e:
                # Try using continuous mode as last resort
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # Try using digital value directly
                        stream_cmd = types.stream_cmd_t(1)  # 1 is usually START_CONTINUOUS
                        stream_cmd.stream_now = True
                        self.rx_streamer.issue_stream_cmd(stream_cmd)
                        logger.debug("Stream start command sent (last resort)")
                    else:
                        logger.error("Failed to send stream start command")
                        return np.array([], dtype=np.complex64)
                except Exception as e2:
                    logger.error(f"Failed to send stream start command: {e2}")
                    return np.array([], dtype=np.complex64)
            
            # Receive data
            total_received = 0
            max_attempts = 3
            attempts = 0
            
            while attempts < max_attempts and total_received < num_samples:
                try:
                    # Receive data with timeout
                    num_rx = self.rx_streamer.recv(buffer[total_received:], metadata, timeout=timeout)
                    
                    # Check reception status
                    error_code = getattr(metadata, 'error_code', None)
                    
                    # Handle errors
                    if error_code and error_code != 0:
                        error_str = str(error_code)
                        if 'overflow' in error_str.lower():
                            logger.warning(f"Overflow error: {error_code}")
                        elif 'sequence' in error_str.lower():
                            logger.warning(f"Out of sequence error: {error_code}")
                        else:
                            logger.warning(f"Reception error: {error_code}")
                        attempts += 1
                        continue
                    
                    # Successful reception
                    if num_rx > 0:
                        total_received += num_rx
                        logger.debug(f"Received {num_rx} samples, total: {total_received}")
                        if total_received >= num_samples:
                            break
                    else:
                        logger.debug("No data received")
                        attempts += 1
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Exception during reception: {e}")
                    attempts += 1
                    time.sleep(0.1)
            
            # Send stream stop command
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # Try using stop mode
                    if hasattr(types.StreamMode, 'stop'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop)
                        self.rx_streamer.issue_stream_cmd(stop_cmd)
                        logger.debug("Stream stop command sent (stop mode)")
                    # Fallback to stop_cont mode
                    elif hasattr(types.StreamMode, 'stop_cont'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop_cont)
                        self.rx_streamer.issue_stream_cmd(stop_cmd)
                        logger.debug("Stream stop command sent (stop_cont mode)")
                else:
                    # Try libpyuhd path
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # Try to find stop mode
                        stop_mode = None
                        # First try continuous mode stop
                        mode_options = ['STREAM_MODE_STOP_CONT', 'stream_mode_stop_cont', 'STOP_CONT', 'STREAM_MODE_STOP_CONTINUOUS', 'stop_continuous']
                        for mode_name in mode_options:
                            if hasattr(types.stream_cmd_t, mode_name):
                                stop_mode = getattr(types.stream_cmd_t, mode_name)
                                break
                        
                        # If continuous mode stop not found, try normal stop
                        if stop_mode is None:
                            mode_options = ['STREAM_MODE_STOP', 'stream_mode_stop', 'STOP']
                            for mode_name in mode_options:
                                if hasattr(types.stream_cmd_t, mode_name):
                                    stop_mode = getattr(types.stream_cmd_t, mode_name)
                                    break
                        
                        if stop_mode:
                            stop_cmd = types.stream_cmd_t(stop_mode)
                            self.rx_streamer.issue_stream_cmd(stop_cmd)
                            logger.debug("Stream stop command sent (libpyuhd)")
                        else:
                            # Try using digital value directly
                            stop_cmd = types.stream_cmd_t(2)  # 2 is usually STOP_CONTINUOUS
                            self.rx_streamer.issue_stream_cmd(stop_cmd)
                            logger.debug("Stream stop command sent (digital value 2)")
            except Exception as e:
                # Try using stop mode as last resort
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # Try using digital value directly
                        stop_cmd = types.stream_cmd_t(2)  # 2 is usually STOP_CONTINUOUS
                        self.rx_streamer.issue_stream_cmd(stop_cmd)
                        logger.debug("Stream stop command sent (last resort)")
                except Exception as e2:
                    logger.error(f"Failed to send stream stop command: {e2}")
            
            # Check result
            if total_received > 0:
                logger.debug(f"Data reception completed, total: {total_received} samples")
                return buffer[:total_received]
            else:
                logger.warning("No valid data received")
                return np.array([], dtype=np.complex64)
            
        except Exception as e:
            logger.error(f"Failed to receive samples: {e}")
            return np.array([], dtype=np.complex64)
        finally:
            # Clean up streamer
            if self.rx_streamer:
                try:
                    # Streamer is automatically cleaned up in Python
                    pass
                except Exception:
                    pass

class SignalProcessor:
    """Signal processing class"""
    
    def __init__(self, sample_rate):
        """Initialize signal processor"""
        self.sample_rate = sample_rate
    
    def estimate_bandwidth(self, samples):
        """Estimate signal bandwidth"""
        try:
            # Calculate power spectral density
            f, Pxx = signal.welch(samples, self.sample_rate, nperseg=1024)
            
            # Find maximum power
            max_power = np.max(Pxx)
            
            # Find frequency range where power is above -10dB
            threshold = max_power * 0.1  # 10dB attenuation
            
            # Find all frequency indices above threshold
            above_threshold = np.where(Pxx >= threshold)[0]
            
            if len(above_threshold) > 0:
                # Calculate bandwidth - consider both positive and negative frequencies
                min_freq = f[above_threshold[0]]
                max_freq = f[above_threshold[-1]]
                bandwidth = 2 * max_freq  # For baseband signals
                logger.debug(f"Estimated bandwidth: {bandwidth/1e6:.3f} MHz")
                return bandwidth
            else:
                logger.debug("No signal detected, returning 0 bandwidth")
                return 0
        except Exception as e:
            logger.error(f"Failed to estimate bandwidth: {e}")
            return 0
    
    def calculate_snr(self, samples):
        """Calculate signal-to-noise ratio"""
        try:
            # Calculate signal power
            signal_power = np.mean(np.abs(samples)**2)
            
            # Estimate noise through spectrum analysis
            f, Pxx = signal.welch(samples, self.sample_rate, nperseg=1024)
            
            # Find noise bands outside the signal
            noise_bands = np.concatenate([Pxx[:50], Pxx[-50:]])
            noise_power = np.mean(noise_bands)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                logger.debug(f"Calculated SNR: {snr:.2f} dB")
                return snr
            else:
                logger.debug("Noise power too low, returning 0 SNR")
                return 0
        except Exception as e:
            logger.error(f"Failed to calculate SNR: {e}")
            return 0
    
    def analyze_constellation(self, samples, num_points=1000):
        """Analyze constellation for FM characteristics"""
        try:
            # Take first N points for analysis
            analysis_samples = samples[:num_points]
            
            # Calculate constellation points
            i = np.real(analysis_samples)
            q = np.imag(analysis_samples)
            
            # Calculate IQ point standard deviations
            i_std = np.std(i)
            q_std = np.std(q)
            
            # Calculate IQ point circularity (FM signals should be circular)
            circularity = abs(i_std - q_std) / max(i_std, q_std, 1e-10)
            
            # Calculate phase variation
            phase = np.angle(analysis_samples)
            phase_diff = np.diff(phase)
            phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
            phase_variation = np.std(phase_diff)
            
            # Calculate signal power
            power = np.mean(np.abs(analysis_samples)**2)
            
            # Calculate magnitude stability (FM signals should have stable magnitude)
            magnitude = np.abs(analysis_samples)
            mag_mean = np.mean(magnitude)
            mag_std = np.std(magnitude)
            mag_stability = mag_std / mag_mean
            
            # Determine if signal is FM-like
            is_fm_like = False
            if (circularity < 0.5 and  # Circular constellation
                0.05 < phase_variation < 4.0 and  # Phase variation
                mag_stability < 1.0 and  # Stable magnitude
                power > 0.0001):  # Sufficient power
                is_fm_like = True
            
            logger.debug(f"Constellation analysis: circularity={circularity:.3f}, phase_variation={phase_variation:.3f}, mag_stability={mag_stability:.3f}, power={power:.3f}, is_fm_like={is_fm_like}")
            
            return {
                'circularity': circularity,
                'phase_variation': phase_variation,
                'mag_stability': mag_stability,
                'power': power,
                'is_fm_like': is_fm_like
            }
        except Exception as e:
            logger.error(f"Failed to analyze constellation: {e}")
            return {
                'circularity': 0,
                'phase_variation': 0,
                'mag_stability': 0,
                'power': 0,
                'is_fm_like': False
            }
    
    def fm_demod(self, samples):
        """FM demodulation using phase difference"""
        try:
            # Calculate phase difference
            phase = np.angle(samples)
            phase_diff = np.diff(phase)
            
            # Handle phase jumps
            phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
            
            # Normalize output
            max_abs = np.max(np.abs(phase_diff))
            if max_abs > 0:
                demodulated = phase_diff / max_abs
            else:
                demodulated = phase_diff
            
            logger.debug(f"FM demodulation completed, samples: {len(demodulated)}")
            return demodulated
        except Exception as e:
            logger.error(f"Failed to demodulate FM: {e}")
            return np.array([])
    
    def detect_fm_video_sync(self, demodulated, sync_rate=15625):
        """Detect video sync signal in demodulated FM"""
        try:
            # Calculate power spectrum
            f, Pxx = signal.welch(demodulated, self.sample_rate, nperseg=4096)
            
            # Find index of sync frequency
            sync_freq_idx = np.argmin(np.abs(f - sync_rate))
            
            # Check if sync frequency power is significantly higher than surrounding frequencies
            if sync_freq_idx > 0 and sync_freq_idx < len(f) - 1:
                # Calculate average power around sync frequency
                window = 10
                start = max(0, sync_freq_idx - window)
                end = min(len(f), sync_freq_idx + window)
                
                # Exclude sync frequency itself
                mask = np.ones(end - start, dtype=bool)
                mask[sync_freq_idx - start] = False
                
                avg_power = np.mean(Pxx[start:end][mask])
                sync_power = Pxx[sync_freq_idx]
                
                # If sync power is 3x higher than average, consider it a sync signal
                if sync_power > 3 * avg_power:
                    logger.debug(f"Video sync signal detected at {sync_rate} Hz")
                    return True
            
            logger.debug("No video sync signal detected")
            return False
        except Exception as e:
            logger.error(f"Failed to detect video sync: {e}")
            return False

class SignalDetector:
    """Signal detection class"""
    
    def __init__(self, signal_processor):
        """Initialize signal detector"""
        self.signal_processor = signal_processor
    
    def detect_signal(self, samples, freq):
        """Detect and classify signal"""
        try:
            # Estimate bandwidth
            bandwidth = self.signal_processor.estimate_bandwidth(samples)
            
            # Calculate SNR
            snr = self.signal_processor.calculate_snr(samples)
            
            # Analyze constellation
            constellation_info = self.signal_processor.analyze_constellation(samples)
            
            # Basic signal detection
            is_fm_signal = False
            is_fpv_signal = False
            
            # Bandwidth check for FM (6-9 MHz for FPV, 0.05-0.3 MHz for FM broadcast)
            if 0.05e6 < bandwidth < 0.3e6:
                # FM broadcast
                is_fm_signal = constellation_info['is_fm_like'] and snr > 10
            elif 6e6 < bandwidth < 9e6:
                # FPV signal
                is_fm_signal = constellation_info['is_fm_like'] and snr > 5
                
                # Demodulate and check for video sync
                if is_fm_signal:
                    demodulated = self.signal_processor.fm_demod(samples)
                    if len(demodulated) > 0:
                        is_fpv_signal = self.signal_processor.detect_fm_video_sync(demodulated)
            
            signal_info = {
                'frequency': freq,
                'bandwidth': bandwidth,
                'snr': snr,
                'is_fm': is_fm_signal,
                'is_fpv': is_fpv_signal,
                'constellation': constellation_info
            }
            
            logger.info(f"Signal detected at {freq/1e6:.2f} MHz: bandwidth={bandwidth/1e6:.2f} MHz, SNR={snr:.2f} dB, FM={is_fm_signal}, FPV={is_fpv_signal}")
            return signal_info
        except Exception as e:
            logger.error(f"Failed to detect signal: {e}")
            return {
                'frequency': freq,
                'bandwidth': 0,
                'snr': 0,
                'is_fm': False,
                'is_fpv': False,
                'constellation': {}
            }

class DataStorage:
    """Data storage class"""
    
    def __init__(self, storage_dir='data'):
        """Initialize data storage"""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_signal_info(self, signal_info):
        """Save signal information to JSON"""
        try:
            # Create filename based on frequency
            freq_mhz = signal_info['frequency'] / 1e6
            filename = f"signal_{freq_mhz:.2f}MHz.json"
            filepath = os.path.join(self.storage_dir, filename)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(signal_info, f, indent=2, default=str)
            
            logger.info(f"Signal info saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save signal info: {e}")
            return False
    
    def save_samples(self, samples, freq):
        """Save raw samples"""
        try:
            # Create filename based on frequency
            freq_mhz = freq / 1e6
            filename = f"samples_{freq_mhz:.2f}MHz.npy"
            filepath = os.path.join(self.storage_dir, filename)
            
            # Save to numpy file
            np.save(filepath, samples)
            
            logger.info(f"Samples saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save samples: {e}")
            return False
    
    def save_demodulated(self, demodulated, freq):
        """Save demodulated signal"""
        try:
            # Create filename based on frequency
            freq_mhz = freq / 1e6
            filename = f"demodulated_{freq_mhz:.2f}MHz.npy"
            filepath = os.path.join(self.storage_dir, filename)
            
            # Save to numpy file
            np.save(filepath, demodulated)
            
            logger.info(f"Demodulated signal saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save demodulated signal: {e}")
            return False

class SignalVisualizer:
    """Signal visualizer class"""
    
    def __init__(self, sample_rate, fft_size=1024, window_size=50):
        """Initialize visualizer"""
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window_size = window_size
        
        # Create figure with multiple subplots
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Industrial FPV Scanner - Real-time Visualization', fontsize=16)
        
        # Main spectrum plot
        self.ax_spectrum = self.fig.add_subplot(2, 1, 1)
        self.ax_spectrum.set_title('Real-time Spectrum')
        self.ax_spectrum.set_xlabel('Frequency (MHz)')
        self.ax_spectrum.set_ylabel('Signal Strength (dB)')
        self.ax_spectrum.grid(True)
        
        # Time-domain plot
        self.ax_time = self.fig.add_subplot(2, 2, 3)
        self.ax_time.set_title('Time-domain Signal')
        self.ax_time.set_xlabel('Time (ms)')
        self.ax_time.set_ylabel('Amplitude')
        self.ax_time.grid(True)
        
        # Scan progress plot
        self.ax_progress = self.fig.add_subplot(2, 2, 4)
        self.ax_progress.set_title('Scan Progress')
        self.ax_progress.set_xlabel('Frequency (MHz)')
        self.ax_progress.set_ylabel('Signal Strength (dB)')
        self.ax_progress.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Initialize data storage
        self.spectrum_data = []
        self.time_data = []
        self.progress_data = {}
        
        # Initialize plot elements
        self.line_spectrum, = self.ax_spectrum.plot([], [], 'b-')
        self.line_time, = self.ax_time.plot([], [], 'g-')
        self.band_lines = {}
        
        # Current state
        self.current_band = None
        self.current_freq = 0
    
    def update_spectrum(self, samples, current_freq):
        """Update spectrum plot"""
        try:
            # Calculate spectrum
            spectrum = np.abs(np.fft.fft(samples[:self.fft_size]))
            spectrum_db = 20 * np.log10(spectrum + 1e-10)
            
            # Calculate frequency axis
            freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate) / 1e6  # Convert to MHz
            center_freq = current_freq / 1e6
            actual_freqs = freqs + center_freq
            
            # Update spectrum data
            self.spectrum_data.append(spectrum_db)
            if len(self.spectrum_data) > self.window_size:
                self.spectrum_data.pop(0)
            
            # Update spectrum plot
            self.line_spectrum.set_data(actual_freqs, spectrum_db)
            
            # Only update axis limits if they haven't been set yet or if data range changes significantly
            if not hasattr(self, 'spectrum_xlim') or not self.spectrum_xlim:
                self.spectrum_xlim = [min(actual_freqs), max(actual_freqs)]
                self.spectrum_ylim = [np.min(spectrum_db) - 5, np.max(spectrum_db) + 5]
            else:
                # Only adjust if new data range is significantly different
                new_min_x = min(actual_freqs)
                new_max_x = max(actual_freqs)
                new_min_y = np.min(spectrum_db) - 5
                new_max_y = np.max(spectrum_db) + 5
                
                # Only update if the change is more than 10%
                if (abs(new_min_x - self.spectrum_xlim[0]) > 0.1 * abs(self.spectrum_xlim[1] - self.spectrum_xlim[0]) or
                    abs(new_max_x - self.spectrum_xlim[1]) > 0.1 * abs(self.spectrum_xlim[1] - self.spectrum_xlim[0]) or
                    abs(new_min_y - self.spectrum_ylim[0]) > 0.1 * abs(self.spectrum_ylim[1] - self.spectrum_ylim[0]) or
                    abs(new_max_y - self.spectrum_ylim[1]) > 0.1 * abs(self.spectrum_ylim[1] - self.spectrum_ylim[0])):
                    self.spectrum_xlim = [new_min_x, new_max_x]
                    self.spectrum_ylim = [new_min_y, new_max_y]
            
            # Set axis limits
            self.ax_spectrum.set_xlim(self.spectrum_xlim)
            self.ax_spectrum.set_ylim(self.spectrum_ylim)
            
            # Update title with current frequency
            self.ax_spectrum.set_title(f'Real-time Spectrum - Current: {center_freq:.2f} MHz')
            
            return [self.line_spectrum]
        except Exception as e:
            logger.error(f"Failed to update spectrum plot: {e}")
            return []
    
    def update_time(self, samples):
        """Update time-domain plot"""
        try:
            # Calculate time axis
            t = np.arange(len(samples[:self.fft_size])) / self.sample_rate * 1000  # Convert to ms
            
            # Use real part of samples for time-domain display
            time_signal = np.real(samples[:self.fft_size])
            
            # Update time data
            self.time_data.append(time_signal)
            if len(self.time_data) > self.window_size:
                self.time_data.pop(0)
            
            # Update time plot
            self.line_time.set_data(t, time_signal)
            
            # Only update axis limits if they haven't been set yet
            if not hasattr(self, 'time_xlim') or not self.time_xlim:
                self.time_xlim = [0, max(t)]
                self.time_ylim = [-1.5, 1.5]  # Fixed range for better stability
            
            # Set axis limits
            self.ax_time.set_xlim(self.time_xlim)
            self.ax_time.set_ylim(self.time_ylim)
            
            return [self.line_time]
        except Exception as e:
            logger.error(f"Failed to update time plot: {e}")
            return []
    
    def update_progress(self, band_name, current_freq, signal_strength):
        """Update scan progress plot"""
        try:
            # Initialize band data if not exists
            if band_name not in self.progress_data:
                self.progress_data[band_name] = []
                # Create line for this band
                line, = self.ax_progress.plot([], [], '-', label=band_name)
                self.band_lines[band_name] = line
                # Update legend
                self.ax_progress.legend()
            
            # Add current data point
            self.progress_data[band_name].append((current_freq / 1e6, signal_strength))
            
            # Limit data points to avoid overcrowding
            max_points = 500  # Reduce points to improve performance
            if len(self.progress_data[band_name]) > max_points:
                self.progress_data[band_name] = self.progress_data[band_name][-max_points:]
            
            # Update progress plot for all bands
            all_freqs = []
            all_strengths = []
            for band in self.progress_data:
                if self.progress_data[band]:
                    freqs, strengths = zip(*self.progress_data[band])
                    self.band_lines[band].set_data(freqs, strengths)
                    all_freqs.extend(freqs)
                    all_strengths.extend(strengths)
            
            # Only update axis limits if they haven't been set yet
            if all_freqs and (not hasattr(self, 'progress_xlim') or not self.progress_xlim):
                self.progress_xlim = [min(all_freqs) - 5, max(all_freqs) + 5]
            if all_strengths and (not hasattr(self, 'progress_ylim') or not self.progress_ylim):
                self.progress_ylim = [min(all_strengths) - 10, max(all_strengths) + 10]
            
            # Set axis limits
            if hasattr(self, 'progress_xlim') and self.progress_xlim:
                self.ax_progress.set_xlim(self.progress_xlim)
            if hasattr(self, 'progress_ylim') and self.progress_ylim:
                self.ax_progress.set_ylim(self.progress_ylim)
            
            return list(self.band_lines.values())
        except Exception as e:
            logger.error(f"Failed to update progress plot: {e}")
            return []
    
    def update(self, samples, current_freq, band_name, signal_strength):
        """Update all plots"""
        try:
            spectrum_artists = self.update_spectrum(samples, current_freq)
            time_artists = self.update_time(samples)
            progress_artists = self.update_progress(band_name, current_freq, signal_strength)
            
            return spectrum_artists + time_artists + progress_artists
        except Exception as e:
            logger.error(f"Failed to update plots: {e}")
            return []

class SystemMonitor:
    """System monitoring class"""
    
    def __init__(self):
        """Initialize system monitor"""
        self.start_time = time.time()
        self.signal_count = 0
        self.fpv_count = 0
        self.error_count = 0
    
    def update(self, signal_info=None, error=None):
        """Update system status"""
        if error:
            self.error_count += 1
            logger.error(f"System error: {error}")
        elif signal_info:
            self.signal_count += 1
            if signal_info.get('is_fpv', False):
                self.fpv_count += 1
    
    def get_status(self):
        """Get system status"""
        uptime = time.time() - self.start_time
        status = {
            'uptime': uptime,
            'signal_count': self.signal_count,
            'fpv_count': self.fpv_count,
            'error_count': self.error_count
        }
        return status
    
    def log_status(self):
        """Log system status"""
        status = self.get_status()
        logger.info(f"System status: uptime={status['uptime']:.2f}s, signals={status['signal_count']}, fpv={status['fpv_count']}, errors={status['error_count']}")

class ConfigManager:
    """Configuration management class"""
    
    def __init__(self, config_file='config.json'):
        """Initialize config manager"""
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
                return config
            else:
                # Default configuration
                default_config = {
                    'sample_rate': 2e6,
                    'gain': 30,
                    'antenna': 'RX2',
                    'bands': [
                        {
                            'name': 'FM Broadcast',
                            'start': 87.5e6,
                            'stop': 108e6,
                            'step': 0.1e6
                        },
                        {
                            'name': '5.8G FPV',
                            'start': 5725e6,
                            'stop': 5925e6,
                            'step': 1e6
                        }
                    ],
                    'scan_params': {
                        'num_samples': 8192,
                        'timeout': 1.0,
                        'continuous': True
                    },
                    'visualization': {
                        'fft_size': 1024,
                        'window_size': 50,
                        'update_interval': 50
                    }
                }
                logger.info("Using default configuration")
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")

class IndustrialFPVScanner:
    """Industrial-grade FPV scanner"""
    
    def __init__(self):
        """Initialize scanner"""
        self.config_manager = ConfigManager()
        self.device_manager = DeviceManager()
        self.data_acquisition = None
        self.signal_processor = None
        self.signal_detector = None
        self.data_storage = DataStorage()
        self.visualizer = None
        self.system_monitor = SystemMonitor()
        self.is_running = False
        self.scan_thread = None
        self.data_queue = queue.Queue()
    
    def initialize(self):
        """Initialize all components"""
        try:
            # Load configuration
            sample_rate = self.config_manager.get('sample_rate', 2e6)
            gain = self.config_manager.get('gain', 30)
            antenna = self.config_manager.get('antenna', 'RX2')
            
            # Initialize device
            if not self.device_manager.initialize(sample_rate, gain, antenna):
                logger.error("Failed to initialize device")
                return False
            
            # Initialize components
            self.data_acquisition = DataAcquisition(self.device_manager)
            self.signal_processor = SignalProcessor(sample_rate)
            self.signal_detector = SignalDetector(self.signal_processor)
            
            # Initialize visualizer
            fft_size = self.config_manager.get('visualization.fft_size', 1024)
            window_size = self.config_manager.get('visualization.window_size', 50)
            self.visualizer = SignalVisualizer(sample_rate, fft_size, window_size)
            
            logger.info("Scanner initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize scanner: {e}")
            return False
    
    def scan_band(self, band):
        """Scan a single band"""
        try:
            current_freq = band['start']
            while current_freq <= band['stop'] and self.is_running:
                # Set frequency
                if not self.device_manager.set_frequency(current_freq):
                    logger.error(f"Failed to set frequency: {current_freq/1e6:.2f} MHz")
                    current_freq += band['step']
                    continue
                
                # Wait for frequency to stabilize
                time.sleep(0.1)
                
                # Receive samples
                num_samples = self.config_manager.get('scan_params.num_samples', 8192)
                timeout = self.config_manager.get('scan_params.timeout', 1.0)
                samples = self.data_acquisition.receive_samples(num_samples, timeout)
                
                if len(samples) > 0:
                    # Calculate signal strength
                    signal_strength = 20 * np.log10(np.mean(np.abs(samples)**2) + 1e-10)
                    
                    # Detect signal
                    signal_info = self.signal_detector.detect_signal(samples, current_freq)
                    
                    # Store data
                    self.data_storage.save_signal_info(signal_info)
                    if signal_info['is_fm']:
                        self.data_storage.save_samples(samples, current_freq)
                        demodulated = self.signal_processor.fm_demod(samples)
                        if len(demodulated) > 0:
                            self.data_storage.save_demodulated(demodulated, current_freq)
                    
                    # Update system monitor
                    self.system_monitor.update(signal_info)
                    
                    # Add to data queue for visualization
                    self.data_queue.put((samples, current_freq, band['name'], signal_strength))
                
                # Move to next frequency
                current_freq += band['step']
        except Exception as e:
            logger.error(f"Error scanning band {band['name']}: {e}")
            self.system_monitor.update(error=str(e))
    
    def scan_thread_func(self):
        """Scan thread function"""
        try:
            bands = self.config_manager.get('bands', [])
            continuous = self.config_manager.get('scan_params.continuous', True)
            
            while self.is_running:
                for band in bands:
                    if not self.is_running:
                        break
                    logger.info(f"Starting scan of band: {band['name']} ({band['start']/1e6:.2f}-{band['stop']/1e6:.2f} MHz)")
                    self.scan_band(band)
                
                if not continuous:
                    break
                
                # Wait before next scan cycle
                logger.info("Scan cycle completed, waiting for next cycle...")
                time.sleep(5)
        except Exception as e:
            logger.error(f"Error in scan thread: {e}")
            self.system_monitor.update(error=str(e))
    
    def start_scan(self):
        """Start scanning"""
        try:
            self.is_running = True
            self.scan_thread = threading.Thread(target=self.scan_thread_func)
            self.scan_thread.daemon = True
            self.scan_thread.start()
            logger.info("Scan started")
            return True
        except Exception as e:
            logger.error(f"Failed to start scan: {e}")
            return False
    
    def stop_scan(self):
        """Stop scanning"""
        try:
            self.is_running = False
            if self.scan_thread:
                self.scan_thread.join(timeout=5.0)
            logger.info("Scan stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop scan: {e}")
            return False
    
    def update_visualization(self, frame):
        """Update visualization"""
        try:
            # Process data from queue
            artists = []
            while not self.data_queue.empty():
                try:
                    samples, current_freq, band_name, signal_strength = self.data_queue.get(block=False)
                    artists.extend(self.visualizer.update(samples, current_freq, band_name, signal_strength))
                except queue.Empty:
                    break
            
            # Log system status periodically
            if frame % 100 == 0:
                self.system_monitor.log_status()
            
            return artists
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            return []
    
    def run(self):
        """Run scanner with visualization"""
        try:
            # Start scanning
            if not self.start_scan():
                logger.error("Failed to start scanning")
                return
            
            # Start visualization
            update_interval = self.config_manager.get('visualization.update_interval', 50)
            ani = FuncAnimation(
                self.visualizer.fig,
                self.update_visualization,
                interval=update_interval,
                blit=True
            )
            
            # Display figure
            plt.show()
        except KeyboardInterrupt:
            logger.info("User interrupted")
        finally:
            # Stop scanning
            self.stop_scan()
            # Close device
            self.device_manager.close()
            logger.info("Scanner stopped and resources released")

def main():
    """Main function"""
    logger.info("Industrial FPV Scanner started")
    
    # Create scanner instance
    scanner = IndustrialFPVScanner()
    
    # Initialize scanner
    if not scanner.initialize():
        logger.error("Failed to initialize scanner")
        sys.exit(1)
    
    # Run scanner
    scanner.run()
    
    logger.info("Industrial FPV Scanner ended")

if __name__ == "__main__":
    main()
