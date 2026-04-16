#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业级FM信号处理系统
功能：FM信号搜集、标注、解调、还原与可视化
设备：USRP B210 / B200mini / 其他SDR设备
技术栈：Python 3.8-3.11 + uhd + numpy + scipy + matplotlib + pyaudio
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import uhd
import json
import os
from datetime import datetime

# 尝试导入 pyaudio 用于音频输出
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("pyaudio 未安装，音频输出功能将不可用")

# 尝试导入 scipy.io.wavfile 用于音频保存
try:
    import scipy.io.wavfile as wav
    WAV_AVAILABLE = True
except ImportError:
    WAV_AVAILABLE = False
    print("scipy.io 未安装，音频保存功能将不可用")

class ConfigManager:
    """Configuration management class"""
    
    def __init__(self, config_file='fm_processor_config.json'):
        """Initialize configuration"""
        self.config_file = config_file
        self.default_config = {
            # Device parameters
            'device': {
                'sample_rate': 2e6,  # Sample rate (2 MHz)
                'gain': 30,          # Gain (30 dB)
                'antenna': "RX2"      # Antenna
            },
            # Scan parameters
            'scan': {
                'bands': [
                    # FM broadcast band
                    {
                        'name': 'FM Broadcast',
                        'start': 87.5e6,
                        'stop': 108e6,
                        'step': 0.1e6,
                        'color': 'blue'
                    },
                    # 5.8G FPV band
                    {
                        'name': '5.8G FPV',
                        'start': 5725e6,
                        'stop': 5925e6,
                        'step': 1e6,
                        'color': 'red'
                    }
                ],
                'signal_detection': {
                    'snr_threshold': 10,      # SNR threshold
                    'bandwidth_min': 0.05e6,  # Minimum bandwidth
                    'bandwidth_max': 0.3e6    # Maximum bandwidth
                }
            },
            # Demodulation parameters
            'demodulation': {
                'deemphasis_tau': 75e-6,    # De-emphasis time constant
                'audio_cutoff': 15000,      # Audio cutoff frequency
                'audio_sample_rate': 48000  # Audio sample rate
            },
            # Visualization parameters
            'visualization': {
                'update_interval': 50,      # Update interval (ms)
                'fft_size': 1024,           # FFT size
                'window_size': 50,          # Number of time steps to display
                'figsize': [16, 12]         # Figure size
            },
            # Output parameters
            'output': {
                'save_audio': True,         # Save audio
                'save_metadata': True,      # Save metadata
                'output_dir': 'output'      # Output directory
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with default configuration
                return self._merge_configs(self.default_config, config)
            else:
                print(f"Configuration file {self.config_file} does not exist, using default configuration")
                self.save_config(self.default_config)
                return self.default_config
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            return self.default_config
    
    def save_config(self, config):
        """Save configuration file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Failed to save configuration file: {e}")
    
    def _merge_configs(self, default, user):
        """Merge configurations"""
        if isinstance(default, dict) and isinstance(user, dict):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    default[key] = self._merge_configs(default[key], value)
                else:
                    default[key] = value
        return default

class SignalProcessor:
    """Signal processing class"""
    
    def __init__(self, config):
        """Initialize signal processor"""
        self.config = config
        self.device_config = config['device']
        self.scan_config = config['scan']
        self.demod_config = config['demodulation']
        self.output_config = config['output']
        
        # Create output directory
        self.output_dir = self.output_config['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Store detected signals
        self.detected_signals = []
        
    def estimate_bandwidth(self, samples, sample_rate):
        """Estimate signal bandwidth"""
        # Calculate power spectral density
        f, Pxx = signal.welch(samples, sample_rate, nperseg=1024)
        
        # Find maximum power
        max_power = np.max(Pxx)
        
        # Find frequency range where power is 10dB below maximum
        threshold = max_power * 0.1  # 10dB attenuation
        
        # Find all frequency indices above threshold
        above_threshold = np.where(Pxx >= threshold)[0]
        
        if len(above_threshold) > 0:
            # Calculate bandwidth - considering positive and negative frequencies
            min_freq = f[above_threshold[0]]
            max_freq = f[above_threshold[-1]]
            bandwidth = 2 * max_freq
            return bandwidth
        else:
            return 0
    
    def calculate_snr(self, samples):
        """Calculate signal-to-noise ratio"""
        # Calculate signal power
        signal_power = np.mean(np.abs(samples)**2)
        
        # Estimate noise by analyzing spectrum outside signal band
        f, Pxx = signal.welch(samples, self.device_config['sample_rate'], nperseg=1024)
        
        # Find noise outside signal band
        noise_bands = np.concatenate([Pxx[:50], Pxx[-50:]])
        noise_power = np.mean(noise_bands)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0
        
        return snr
    
    def fm_demod(self, samples):
        """FM demodulation using phase difference method"""
        # Calculate phase difference
        phase = np.angle(samples)
        phase_diff = np.diff(phase)
        
        # Handle phase wraps
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        
        # Normalize output
        max_abs = np.max(np.abs(phase_diff))
        if max_abs > 0:
            demodulated = phase_diff / max_abs
        else:
            demodulated = phase_diff
        
        return demodulated
    
    def classify_signal(self, demodulated):
        """Classify signal type"""
        # Calculate power spectrum
        f, Pxx = signal.welch(demodulated, self.device_config['sample_rate'], nperseg=4096)
        
        # Detect audio spectrum features
        audio_band = (f > 20) & (f < 20000)
        audio_power = np.mean(Pxx[audio_band])
        
        # Detect voice spectrum features (300-3400 Hz)
        voice_band = (f > 300) & (f < 3400)
        voice_power = np.mean(Pxx[voice_band])
        
        # Detect music spectrum features (wider frequency range)
        music_band = (f > 20) & (f < 15000)
        music_power = np.mean(Pxx[music_band])
        
        # Classification decision
        if audio_power > 0.1 * np.max(Pxx):
            if voice_power > 0.5 * audio_power:
                return "VOICE"
            elif music_power > 0.6 * audio_power:
                return "MUSIC"
            else:
                return "AUDIO"
        else:
            return "UNKNOWN"
    
    def restore_audio(self, demodulated, frequency):
        """Restore audio content"""
        if not WAV_AVAILABLE:
            print("  Audio saving functionality not available")
            return None
        
        try:
            # 1. De-emphasis (FM broadcast uses 75 μs de-emphasis)
            tau = self.demod_config['deemphasis_tau']
            sample_rate = self.device_config['sample_rate']
            alpha = 1.0 / (1.0 + 2 * np.pi * sample_rate * tau)
            b = [alpha]
            a = [1, -(1 - alpha)]
            audio = signal.lfilter(b, a, demodulated)
            
            # 2. Low-pass filtering (limit audio bandwidth)
            nyquist = sample_rate / 2
            cutoff = self.demod_config['audio_cutoff']
            b, a = signal.butter(5, cutoff / nyquist, btype='low')
            audio = signal.lfilter(b, a, audio)
            
            # 3. Resample to standard audio sample rate
            audio_rate = self.demod_config['audio_sample_rate']
            audio = signal.resample(audio, int(len(audio) * audio_rate / sample_rate))
            
            # 4. Normalize
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                audio = audio / max_abs
            
            # 5. Save as WAV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"audio_{frequency/1e6:.2f}MHz_{timestamp}.wav")
            wav.write(output_file, audio_rate, (audio * 32767).astype(np.int16))
            
            print(f"  Audio saved to: {output_file}")
            
            return audio
        except Exception as e:
            print(f"  Audio restoration failed: {e}")
            return None
    
    def save_signal_metadata(self, signal_info):
        """Save signal metadata"""
        if not self.output_config['save_metadata']:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_file = os.path.join(self.output_dir, f"signal_metadata_{timestamp}.json")
            
            # Convert frequencies to MHz for better readability
            for signal in signal_info:
                signal['frequency_mhz'] = signal['frequency'] / 1e6
                signal['bandwidth_mhz'] = signal['bandwidth'] / 1e6
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(signal_info, f, indent=2, ensure_ascii=False)
            
            print(f"  Signal metadata saved to: {metadata_file}")
        except Exception as e:
            print(f"  Failed to save signal metadata: {e}")

class USRP_Device:
    """USRP device class"""
    
    def __init__(self, config):
        """Initialize USRP device"""
        self.config = config['device']
        self.usrp = None
        self.initialize_device()
    
    def initialize_device(self):
        """Initialize USRP device"""
        print("Initializing USRP device...")
        try:
            # Create USRP device
            self.usrp = uhd.usrp.MultiUSRP()
            
            # Configure basic parameters
            self.sample_rate = self.config['sample_rate']
            self.gain = self.config['gain']
            self.antenna = self.config['antenna']
            
            # Set sample rate
            print(f"Setting sample rate: {self.sample_rate / 1e6} MHz")
            self.usrp.set_rx_rate(self.sample_rate, 0)
            actual_rate = self.usrp.get_rx_rate(0)
            print(f"Actual sample rate: {actual_rate / 1e6} MHz")
            
            # Set gain
            print(f"Setting gain: {self.gain} dB")
            self.usrp.set_rx_gain(self.gain, 0)
            actual_gain = self.usrp.get_rx_gain(0)
            print(f"Actual gain: {actual_gain} dB")
            
            # Set antenna
            print(f"Setting antenna: {self.antenna}")
            self.usrp.set_rx_antenna(self.antenna, 0)
            actual_antenna = self.usrp.get_rx_antenna(0)
            print(f"Actual antenna: {actual_antenna}")
            
            # Print detailed device info
            print(f"USRP initialization successful: {self.usrp.get_pp_string()}")
            
        except Exception as e:
            print(f"USRP initialization failed: {e}")
            self.usrp = None
    
    def set_frequency(self, freq):
        """Set frequency"""
        if not self.usrp:
            return False
        
        try:
            # Try different API paths
            try:
                # Try using uhd.libpyuhd.types.tune_request
                if hasattr(uhd, 'libpyuhd') and hasattr(uhd.libpyuhd, 'types') and hasattr(uhd.libpyuhd.types, 'tune_request'):
                    tune_req = uhd.libpyuhd.types.tune_request(freq)
                    self.usrp.set_rx_freq(tune_req, 0)
                # Try using uhd.types.tune_request
                elif hasattr(uhd, 'types') and hasattr(uhd.types, 'tune_request'):
                    tune_req = uhd.types.tune_request(freq)
                    self.usrp.set_rx_freq(tune_req, 0)
                else:
                    # Try setting frequency directly
                    self.usrp.set_rx_freq(freq, 0)
            except Exception as e:
                print(f"Failed to set frequency: {e}")
                return False
            
            return True
        except Exception as e:
            print(f"Failed to set frequency: {e}")
            return False
    
    def receive_samples(self, num_samples=8192):
        """Receive USRP data"""
        if not self.usrp:
            return np.array([], dtype=np.complex64)
        
        try:
            # Construct stream_args
            stream_args = None
            
            # Priority use official recommended method
            try:
                if hasattr(uhd.usrp, 'StreamArgs'):
                    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
                    stream_args.channels = [0]
                else:
                    # Try dictionary format
                    stream_args = {
                        "cpu_format": "fc32",
                        "otw_format": "sc16",
                        "channels": [0]
                    }
            except Exception:
                return np.array([], dtype=np.complex64)
            
            # Create receive stream
            try:
                rx_streamer = self.usrp.get_rx_stream(stream_args)
            except Exception:
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
                else:
                    # Try other methods
                    from uhd.libpyuhd import types
                    if hasattr(types, 'rx_metadata_t'):
                        metadata = types.rx_metadata_t()
                    elif hasattr(types, 'rx_metadata'):
                        metadata = types.rx_metadata()
                    else:
                        return np.array([], dtype=np.complex64)
            except Exception:
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
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    # Fallback to start_cont mode
                    elif hasattr(types.StreamMode, 'start_cont'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start_cont)
                        stream_cmd.stream_now = True
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    else:
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
                            rx_streamer.issue_stream_cmd(stream_cmd)
                        else:
                            # Try using digital value directly
                            stream_cmd = types.stream_cmd_t(1)  # 1 is usually START_CONTINUOUS
                            stream_cmd.stream_now = True
                            rx_streamer.issue_stream_cmd(stream_cmd)
                    else:
                        return np.array([], dtype=np.complex64)
            except Exception:
                # Try using continuous mode as last resort
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # Try using digital value directly
                        stream_cmd = types.stream_cmd_t(1)  # 1 is usually START_CONTINUOUS
                        stream_cmd.stream_now = True
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    else:
                        return np.array([], dtype=np.complex64)
                except Exception:
                    return np.array([], dtype=np.complex64)
            
            # Receive data
            total_received = 0
            max_attempts = 3
            attempts = 0
            
            while attempts < max_attempts and total_received < num_samples:
                try:
                    # Receive data with longer timeout
                    num_rx = rx_streamer.recv(buffer[total_received:], metadata, timeout=1.0)
                    
                    # Check reception status
                    error_code = getattr(metadata, 'error_code', None)
                    
                    # Handle errors
                    if error_code and error_code != 0:
                        attempts += 1
                        continue
                    
                    # Successful reception
                    if num_rx > 0:
                        total_received += num_rx
                        if total_received >= num_samples:
                            break
                    else:
                        attempts += 1
                        time.sleep(0.1)
                        
                except Exception:
                    attempts += 1
                    time.sleep(0.1)
            
            # Send stream stop command
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # Try using stop mode
                    if hasattr(types.StreamMode, 'stop'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop)
                        rx_streamer.issue_stream_cmd(stop_cmd)
                    # Fallback to stop_cont mode
                    elif hasattr(types.StreamMode, 'stop_cont'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop_cont)
                        rx_streamer.issue_stream_cmd(stop_cmd)
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
                            rx_streamer.issue_stream_cmd(stop_cmd)
                        else:
                            # Try using digital value directly
                            stop_cmd = types.stream_cmd_t(2)  # 2 is usually STOP_CONTINUOUS
                            rx_streamer.issue_stream_cmd(stop_cmd)
            except Exception:
                # Try using stop mode as last resort
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # Try using digital value directly
                        stop_cmd = types.stream_cmd_t(2)  # 2 is usually STOP_CONTINUOUS
                        rx_streamer.issue_stream_cmd(stop_cmd)
                except Exception:
                    pass
            
            # Check result
            if total_received > 0:
                return buffer[:total_received]
            else:
                return np.array([], dtype=np.complex64)
            
        except Exception:
            return np.array([], dtype=np.complex64)

class SignalVisualizer:
    """Signal visualization class"""
    
    def __init__(self, config):
        """Initialize visualizer"""
        self.config = config['visualization']
        self.sample_rate = config['device']['sample_rate']
        self.bands = config['scan']['bands']
        
        # Create figure with better layout
        self.fig = plt.figure(figsize=tuple(self.config['figsize']))
        self.fig.suptitle('FM Signal Processor - Real-time Visualization', fontsize=16)
        
        # Main spectrum plot (larger size)
        self.ax_spectrum = self.fig.add_subplot(3, 2, 1)
        self.ax_spectrum.set_title('Real-time Spectrum')
        self.ax_spectrum.set_xlabel('Frequency (MHz)')
        self.ax_spectrum.set_ylabel('Signal Strength (dB)')
        self.ax_spectrum.grid(True)
        
        # Time-domain plot
        self.ax_time = self.fig.add_subplot(3, 2, 3)
        self.ax_time.set_title('Time-domain Signal')
        self.ax_time.set_xlabel('Time (ms)')
        self.ax_time.set_ylabel('Amplitude')
        self.ax_time.grid(True)
        
        # Scan progress plot
        self.ax_progress = self.fig.add_subplot(3, 2, 4)
        self.ax_progress.set_title('Scan Progress')
        self.ax_progress.set_xlabel('Frequency (MHz)')
        self.ax_progress.set_ylabel('Signal Strength (dB)')
        self.ax_progress.grid(True)
        
        # Detected signals plot
        self.ax_signals = self.fig.add_subplot(3, 2, 5)
        self.ax_signals.set_title('Detected Signals')
        self.ax_signals.set_xlabel('Frequency (MHz)')
        self.ax_signals.set_ylabel('Signal Strength (dB)')
        self.ax_signals.grid(True)
        
        # System info and band status
        self.ax_info = self.fig.add_subplot(3, 2, 6)
        self.ax_info.set_title('System Status')
        self.ax_info.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Initialize data storage
        self.spectrum_data = []
        self.time_data = []
        self.progress_data = {}
        self.signal_data = []
        for band in self.bands:
            self.progress_data[band['name']] = []
        
        # Initialize plot elements
        self.line_spectrum, = self.ax_spectrum.plot([], [], 'b-')
        self.line_time, = self.ax_time.plot([], [], 'g-')
        self.band_lines = {}
        for band in self.bands:
            line, = self.ax_progress.plot([], [], '-', color=band['color'], label=band['name'])
            self.band_lines[band['name']] = line
        
        # Add legend
        self.ax_progress.legend()
        
        # Current state
        self.current_band_index = 0
        self.current_freq = self.bands[0]['start']
        self.scan_count = 0
        self.detected_signals = []
    
    def update_spectrum(self, samples, current_freq):
        """Update spectrum plot"""
        # Calculate spectrum
        spectrum = np.abs(np.fft.fft(samples[:self.config['fft_size']))       
        spectrum_db = 20 * np.log10(spectrum + 1e-10)

        # Calculate frequency axis
        freqs = np.fft.fftfreq(self.config['fft_size'], 1/self.sample_rate) / 1e6  # Convert to MHz
        center_freq = current_freq / 1e6
        actual_freqs = freqs + center_freq

        # Update spectrum data
        self.spectrum_data.append(spectrum_db)
        if len(self.spectrum_data) > self.config['window_size']:
            self.spectrum_data.pop(0)

        # Update spectrum plot
        self.line_spectrum.set_data(actual_freqs, spectrum_db)
        self.ax_spectrum.set_xlim(min(actual_freqs), max(actual_freqs))
        self.ax_spectrum.set_ylim(np.min(spectrum_db) - 5, np.max(spectrum_db) + 5)

        # Update title
        self.ax_spectrum.set_title(f'Real-time Spectrum - Current: {center_freq:.2f} MHz')

        # Return both line and title to ensure title is updated with blit
        return [self.line_spectrum, self.ax_spectrum.title]
    
    def update_time(self, samples):
        """Update time-domain plot"""
        # Calculate time axis
        t = np.arange(len(samples[:self.config['fft_size']])) / self.sample_rate * 1000  # Convert to ms
        
        # Use real part of samples for time-domain display
        time_signal = np.real(samples[:self.config['fft_size']])
        
        # Update time data
        self.time_data.append(time_signal)
        if len(self.time_data) > self.config['window_size']:
            self.time_data.pop(0)
        
        # Update time plot
        self.line_time.set_data(t, time_signal)
        self.ax_time.set_xlim(0, max(t))
        self.ax_time.set_ylim(np.min(time_signal) - 0.1, np.max(time_signal) + 0.1)
        
        return [self.line_time]
    
    def update_progress(self, band_name, current_freq, signal_strength):
        """Update scan progress plot"""
        # Add current data point
        self.progress_data[band_name].append((current_freq / 1e6, signal_strength))

        # Limit data points
        max_points = 1000
        if len(self.progress_data[band_name]) > max_points:
            self.progress_data[band_name] = self.progress_data[band_name][-max_points:]

        # Only show current band progress for better visualization
        # Clear all lines first
        for band in self.bands:
            if self.band_lines.get(band['name']):
                self.band_lines[band['name']].set_data([], [])

        # Update only current band
        if self.progress_data[band_name]:
            freqs, strengths = zip(*self.progress_data[band_name])
            self.band_lines[band_name].set_data(freqs, strengths)

            # Update axis limits for current band only
            if freqs:
                # Get band information to set appropriate limits
                current_band = None
                for band in self.bands:
                    if band['name'] == band_name:
                        current_band = band
                        break

                if current_band:
                    # Set x-axis limits based on current band range
                    self.ax_progress.set_xlim(current_band['start']/1e6 - 1, current_band['stop']/1e6 + 1)
                else:
                    # Fallback to data range
                    self.ax_progress.set_xlim(min(freqs) - 1, max(freqs) + 1)

            if strengths:
                self.ax_progress.set_ylim(min(strengths) - 5, max(strengths) + 5)

        return list(self.band_lines.values())
    
    def update_signals(self, signals):
        """Update detected signals plot"""
        self.detected_signals = signals
        
        # Clear signals plot
        self.ax_signals.clear()
        self.ax_signals.set_title('Detected Signals')
        self.ax_signals.set_xlabel('Frequency (MHz)')
        self.ax_signals.set_ylabel('Signal Strength (dB)')
        self.ax_signals.grid(True)
        
        # Plot signals
        if signals:
            freqs = [s['frequency'] / 1e6 for s in signals]
            strengths = [s['snr'] for s in signals]
            self.ax_signals.scatter(freqs, strengths, color='red', marker='o')
            
            # Add signal labels
            for i, (freq, strength) in enumerate(zip(freqs, strengths)):
                self.ax_signals.annotate(f"{freq:.2f}MHz", (freq, strength), fontsize=8)
            
            # Set axis limits
            self.ax_signals.set_xlim(min(freqs) - 1, max(freqs) + 1)
            self.ax_signals.set_ylim(min(strengths) - 5, max(strengths) + 5)
        
        return [self.ax_signals]
    
    def update_info(self, current_band, current_freq, scan_count, detected_count):
        """Update system info and band status"""
        self.ax_info.clear()
        self.ax_info.set_title('System Status')
        
        # Current band information
        current_band_name = current_band['name']
        current_band_start = current_band['start'] / 1e6
        current_band_stop = current_band['stop'] / 1e6
        current_freq_mhz = current_freq / 1e6
        
        # System status information
        info_text = [
            f"Current Band: {current_band_name}",
            f"Band Range: {current_band_start:.1f} - {current_band_stop:.1f} MHz",
            f"Current Frequency: {current_freq_mhz:.2f} MHz",
            f"Scan Cycle: {scan_count}",
            f"Detected Signals: {detected_count}",
            f"Sample Rate: {self.sample_rate / 1e6:.1f} MHz"
        ]
        
        # Display band information
        for i, line in enumerate(info_text):
            self.ax_info.text(0.1, 0.8 - i*0.12, line, fontsize=10)
        
        # Display all bands with their ranges
        band_info = "\nBands:\n"
        for band in self.bands:
            band_start = band['start'] / 1e6
            band_stop = band['stop'] / 1e6
            band_info += f"- {band['name']}: {band_start:.1f} - {band_stop:.1f} MHz\n"
        
        self.ax_info.text(0.1, 0.1, band_info, fontsize=9, verticalalignment='top')
        
        return [self.ax_info]
    
    def update_scan(self, scanner, processor):
        """Update scan state"""
        # Get current band
        current_band = self.bands[self.current_band_index]
        
        # Set frequency
        scanner.set_frequency(self.current_freq)
        
        # Receive data
        samples = scanner.receive_samples(num_samples=self.config['fft_size'])
        
        artists = []
        
        if len(samples) == self.config['fft_size']:
            # Calculate signal strength
            signal_strength = processor.calculate_snr(samples)
            
            # Estimate bandwidth
            bandwidth = processor.estimate_bandwidth(samples, self.sample_rate)
            
            # Detect signal
            snr_threshold = self.config.get('signal_detection', {}).get('snr_threshold', 10)
            bandwidth_min = self.config.get('signal_detection', {}).get('bandwidth_min', 0.05e6)
            bandwidth_max = self.config.get('signal_detection', {}).get('bandwidth_max', 0.3e6)
            
            is_fm_signal = False
            if bandwidth_min < bandwidth < bandwidth_max and signal_strength > snr_threshold:
                is_fm_signal = True
                # Check if signal has already been detected
                signal_exists = False
                for sig in self.detected_signals:
                    if abs(sig['frequency'] - self.current_freq) < current_band['step']:
                        signal_exists = True
                        break
                
                if not signal_exists:
                    # Demodulate signal to get type
                    demodulated = processor.fm_demod(samples)
                    signal_type = processor.classify_signal(demodulated)
                    
                    # Add new signal
                    new_signal = {
                        'frequency': self.current_freq,
                        'bandwidth': bandwidth,
                        'snr': signal_strength,
                        'type': signal_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.detected_signals.append(new_signal)
                    print(f"New signal detected: {self.current_freq/1e6:.2f} MHz, Type: {signal_type}")
            
            # Update plots
            spectrum_artists = self.update_spectrum(samples, self.current_freq)
            time_artists = self.update_time(samples)
            progress_artists = self.update_progress(current_band['name'], self.current_freq, signal_strength)
            signal_artists = self.update_signals(self.detected_signals)
            info_artists = self.update_info(current_band, self.current_freq, self.scan_count, len(self.detected_signals))
            
            artists.extend(spectrum_artists)
            artists.extend(time_artists)
            artists.extend(progress_artists)
            artists.extend(signal_artists)
            artists.extend(info_artists)
            
            # Move to next frequency
            self.current_freq += current_band['step']
            
            # Check if band is complete
            if self.current_freq > current_band['stop']:
                # Move to next band
                self.current_band_index = (self.current_band_index + 1) % len(self.bands)
                self.current_freq = self.bands[self.current_band_index]['start']
                self.scan_count += 1
                print(f"Scan cycle {self.scan_count} completed")
        
        return artists

class FM_Industrial_Processor:
    """Industrial FM signal processor"""
    
    def __init__(self):
        """Initialize processor"""
        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        
        # Initialize device
        self.device = USRP_Device(self.config)
        
        # Initialize signal processor
        self.processor = SignalProcessor(self.config)
        
        # Initialize visualizer
        self.visualizer = SignalVisualizer(self.config)
        
    def start_scan(self):
        """Start scanning"""
        print("Starting FM signal scan...")
        
        # Define animation update function
        def update(frame):
            """Update animation"""
            return self.visualizer.update_scan(self.device, self.processor)
        
        # Start animation
        ani = FuncAnimation(
            self.visualizer.fig,
            update,
            interval=self.config['visualization']['update_interval'],
            blit=True
        )
        
        # Display figure
        try:
            plt.show()
        except KeyboardInterrupt:
            print("User interrupted")
        
        # Save detected signals
        if self.visualizer.detected_signals:
            print(f"\nDetected {len(self.visualizer.detected_signals)} FM signals")
            self.processor.save_signal_metadata(self.visualizer.detected_signals)
            
            # Demodulate all detected signals
            try:
                demodulate = input("Demodulate all detected FM signals? (y/n): ")
                if demodulate.lower() == "y":
                    self.demodulate_all_signals()
            except ValueError:
                pass
        else:
            print("No FM signals detected")
    
    def demodulate_all_signals(self):
        """Demodulate all detected signals"""
        signals = self.visualizer.detected_signals
        if not signals:
            print("No FM signals detected, cannot demodulate")
            return
        
        print("\n" + "="*80)
        print("Starting demodulation of all detected FM signals:")
        print("="*80)
        
        for i, signal_info in enumerate(signals):
            frequency = signal_info['frequency']
            print(f"\nDemodulating signal {i+1}: {frequency/1e6:.2f} MHz")
            
            # Configure frequency
            if not self.device.set_frequency(frequency):
                print("  Failed to set frequency, skipping this signal")
                continue
            
            time.sleep(0.1)
            
            # Receive data
            samples = self.device.receive_samples(num_samples=32768)  # More samples for demodulation
            
            if len(samples) == 0:
                print("  No data received, skipping this signal")
                continue
            
            # Demodulate
            demodulated = self.processor.fm_demod(samples)
            
            # Classify
            signal_type = self.processor.classify_signal(demodulated)
            print(f"  Signal type: {signal_type}")
            
            # Restore audio
            if signal_type in ["VOICE", "MUSIC", "AUDIO"]:
                self.processor.restore_audio(demodulated, frequency)
            else:
                print("  Unrecognized signal type, skipping audio restoration")

    def run(self):
        """Run processor"""
        print("Industrial FM Signal Processing System")
        print("=" * 60)
        print(f"Configuration file: {self.config_manager.config_file}")
        print(f"Output directory: {self.config['output']['output_dir']}")
        print("=" * 60)
        
        if not self.device.usrp:
            print("Device initialization failed, cannot continue")
            return
        
        # Start scanning
        self.start_scan()
        
        print("\nScript ended")

if __name__ == "__main__":
    processor = FM_Industrial_Processor()
    processor.run()
