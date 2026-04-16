#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved FM signal scan and visualization script
Function: Real-time FM signal scanning with enhanced visualization
Device: USRP B210 / B200mini
Tech stack: Python 3.8-3.11 + uhd + numpy + scipy + matplotlib
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import uhd

# Configure Matplotlib for better display
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use DejaVu Sans font
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Configurable parameters
# Scan parameters
BANDS = [
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
]

SAMPLE_RATE = 2e6        # Sample rate (2 MHz)
GAIN = 30                # Gain (30 dB)
ANTENNA = "RX2"           # Antenna

# Visualization parameters
UPDATE_INTERVAL = 50     # Update interval (ms)
FFT_SIZE = 1024          # FFT size
WINDOW_SIZE = 50         # Number of time steps to display

class USRP_Scanner:
    """USRP Scanner class"""
    
    def __init__(self):
        """Initialize USRP device"""
        print("Initializing USRP device...")
        try:
            # Create USRP device
            self.usrp = uhd.usrp.MultiUSRP()
            
            # Configure basic parameters
            self.sample_rate = SAMPLE_RATE
            self.gain = GAIN
            self.antenna = ANTENNA
            
            # Set sample rate first
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
            sys.exit(1)
    
    def set_frequency(self, freq):
        """Set frequency, handle different API versions"""
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
    """Signal visualizer class"""
    
    def __init__(self, sample_rate, fft_size=1024, window_size=50):
        """Initialize visualizer"""
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window_size = window_size
        
        # Create figure with multiple subplots
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('FM Signal Scanner - Real-time Visualization', fontsize=16)
        
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
        for band in BANDS:
            self.progress_data[band['name']] = []
        
        # Initialize plot elements
        self.line_spectrum, = self.ax_spectrum.plot([], [], 'b-')
        self.line_time, = self.ax_time.plot([], [], 'g-')
        self.band_lines = {}
        for band in BANDS:
            line, = self.ax_progress.plot([], [], '-', color=band['color'], label=band['name'])
            self.band_lines[band['name']] = line
        
        # Add legend to progress plot
        self.ax_progress.legend()
        
        # Current state
        self.current_band_index = 0
        self.current_freq = BANDS[0]['start']
        self.scan_count = 0
        
    def update_spectrum(self, samples, current_freq):
        """Update spectrum plot"""
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
        self.ax_spectrum.set_xlim(min(actual_freqs), max(actual_freqs))
        self.ax_spectrum.set_ylim(np.min(spectrum_db) - 5, np.max(spectrum_db) + 5)
        
        # Update title with current frequency
        self.ax_spectrum.set_title(f'Real-time Spectrum - Current: {center_freq:.2f} MHz')
        
        return [self.line_spectrum]
    
    def update_time(self, samples):
        """Update time-domain plot"""
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
        self.ax_time.set_xlim(0, max(t))
        self.ax_time.set_ylim(np.min(time_signal) - 0.1, np.max(time_signal) + 0.1)
        
        return [self.line_time]
    
    def update_progress(self, band_name, current_freq, signal_strength):
        """Update scan progress plot"""
        # Add current data point
        self.progress_data[band_name].append((current_freq / 1e6, signal_strength))
        
        # Limit data points to avoid overcrowding
        max_points = 1000
        if len(self.progress_data[band_name]) > max_points:
            self.progress_data[band_name] = self.progress_data[band_name][-max_points:]
        
        # Update progress plot for all bands
        for band in BANDS:
            if self.progress_data[band['name']]:
                freqs, strengths = zip(*self.progress_data[band['name']])
                self.band_lines[band['name']].set_data(freqs, strengths)
                
                # Update axis limits
                all_freqs = []
                all_strengths = []
                for b in BANDS:
                    if self.progress_data[b['name']]:
                        f, s = zip(*self.progress_data[b['name']])
                        all_freqs.extend(f)
                        all_strengths.extend(s)
                
                if all_freqs:
                    self.ax_progress.set_xlim(min(all_freqs) - 1, max(all_freqs) + 1)
                if all_strengths:
                    self.ax_progress.set_ylim(min(all_strengths) - 5, max(all_strengths) + 5)
        
        return list(self.band_lines.values())
    
    def update_scan(self, scanner):
        """Update scan state"""
        # Get current band
        current_band = BANDS[self.current_band_index]
        
        # Set frequency
        scanner.set_frequency(self.current_freq)
        
        # Receive data
        samples = scanner.receive_samples(num_samples=self.fft_size)
        
        if len(samples) == self.fft_size:
            # Calculate signal strength
            signal_strength = 20 * np.log10(np.mean(np.abs(samples)**2) + 1e-10)
            
            # Update plots
            spectrum_artists = self.update_spectrum(samples, self.current_freq)
            time_artists = self.update_time(samples)
            progress_artists = self.update_progress(current_band['name'], self.current_freq, signal_strength)
            
            # Move to next frequency
            self.current_freq += current_band['step']
            
            # Check if band is complete
            if self.current_freq > current_band['stop']:
                # Move to next band
                self.current_band_index = (self.current_band_index + 1) % len(BANDS)
                self.current_freq = BANDS[self.current_band_index]['start']
                self.scan_count += 1
                print(f"Scan cycle {self.scan_count} completed")
            
            return spectrum_artists + time_artists + progress_artists
        
        return []

def main():
    """Main function"""
    print("Improved FM signal scan and visualization")
    print("=" * 60)
    
    # Initialize USRP scanner
    scanner = USRP_Scanner()
    
    # Initialize visualizer
    visualizer = SignalVisualizer(SAMPLE_RATE, FFT_SIZE, WINDOW_SIZE)
    
    # Define animation update function
    def update(frame):
        """Update animation"""
        return visualizer.update_scan(scanner)
    
    # Start animation
    ani = FuncAnimation(
        visualizer.fig,
        update,
        interval=UPDATE_INTERVAL,
        blit=True
    )
    
    # Display figure
    try:
        plt.show()
    except KeyboardInterrupt:
        print("User interrupted")
    
    print("\nScript ended")


if __name__ == "__main__":
    main()
