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
    """配置管理类"""
    
    def __init__(self, config_file='fm_processor_config.json'):
        """初始化配置"""
        self.config_file = config_file
        self.default_config = {
            # 设备参数
            'device': {
                'sample_rate': 2e6,  # 采样率 (2 MHz)
                'gain': 30,          # 增益 (30 dB)
                'antenna': "RX2"      # 天线
            },
            # 扫描参数
            'scan': {
                'bands': [
                    # FM广播频段
                    {
                        'name': 'FM Broadcast',
                        'start': 87.5e6,
                        'stop': 108e6,
                        'step': 0.1e6,
                        'color': 'blue'
                    },
                    # 5.8G FPV频段
                    {
                        'name': '5.8G FPV',
                        'start': 5725e6,
                        'stop': 5925e6,
                        'step': 1e6,
                        'color': 'red'
                    }
                ],
                'signal_detection': {
                    'snr_threshold': 10,      # 信噪比阈值
                    'bandwidth_min': 0.05e6,  # 最小带宽
                    'bandwidth_max': 0.3e6    # 最大带宽
                }
            },
            # 解调参数
            'demodulation': {
                'deemphasis_tau': 75e-6,    # 去加重时间常数
                'audio_cutoff': 15000,      # 音频截止频率
                'audio_sample_rate': 48000  # 音频采样率
            },
            # 可视化参数
            'visualization': {
                'update_interval': 50,      # 更新间隔 (ms)
                'fft_size': 1024,           # FFT大小
                'window_size': 50,          # 显示的时间步数
                'figsize': [16, 12]         # 图表大小
            },
            # 输出参数
            'output': {
                'save_audio': True,         # 保存音频
                'save_metadata': True,      # 保存元数据
                'output_dir': 'output'      # 输出目录
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置
                return self._merge_configs(self.default_config, config)
            else:
                print(f"配置文件 {self.config_file} 不存在，使用默认配置")
                self.save_config(self.default_config)
                return self.default_config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self.default_config
    
    def save_config(self, config):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def _merge_configs(self, default, user):
        """合并配置"""
        if isinstance(default, dict) and isinstance(user, dict):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    default[key] = self._merge_configs(default[key], value)
                else:
                    default[key] = value
        return default

class SignalProcessor:
    """信号处理类"""
    
    def __init__(self, config):
        """初始化信号处理器"""
        self.config = config
        self.device_config = config['device']
        self.scan_config = config['scan']
        self.demod_config = config['demodulation']
        self.output_config = config['output']
        
        # 创建输出目录
        self.output_dir = self.output_config['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 存储检测到的信号
        self.detected_signals = []
        
    def estimate_bandwidth(self, samples, sample_rate):
        """估计信号带宽"""
        # 计算功率谱密度
        f, Pxx = signal.welch(samples, sample_rate, nperseg=1024)
        
        # 找到最大功率
        max_power = np.max(Pxx)
        
        # 找到功率下降 10dB 的频率范围
        threshold = max_power * 0.1  # 10dB 衰减
        
        # 找到所有超过阈值的频率索引
        above_threshold = np.where(Pxx >= threshold)[0]
        
        if len(above_threshold) > 0:
            # 计算带宽 - 考虑正负频率
            min_freq = f[above_threshold[0]]
            max_freq = f[above_threshold[-1]]
            bandwidth = 2 * max_freq
            return bandwidth
        else:
            return 0
    
    def calculate_snr(self, samples):
        """计算信噪比"""
        # 计算信号功率
        signal_power = np.mean(np.abs(samples)**2)
        
        # 假设噪声是信号的一部分，通过频谱分析估计噪声
        f, Pxx = signal.welch(samples, self.device_config['sample_rate'], nperseg=1024)
        
        # 找到信号频段外的噪声
        noise_bands = np.concatenate([Pxx[:50], Pxx[-50:]])
        noise_power = np.mean(noise_bands)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0
        
        return snr
    
    def fm_demod(self, samples):
        """相位差分法 FM 解调"""
        # 计算相位差
        phase = np.angle(samples)
        phase_diff = np.diff(phase)
        
        # 处理相位跳变
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        
        # 归一化输出
        max_abs = np.max(np.abs(phase_diff))
        if max_abs > 0:
            demodulated = phase_diff / max_abs
        else:
            demodulated = phase_diff
        
        return demodulated
    
    def classify_signal(self, demodulated):
        """信号分类"""
        # 计算功率谱
        f, Pxx = signal.welch(demodulated, self.device_config['sample_rate'], nperseg=4096)
        
        # 检测音频频谱特征
        audio_band = (f > 20) & (f < 20000)
        audio_power = np.mean(Pxx[audio_band])
        
        # 检测语音频谱特征 (300-3400 Hz)
        voice_band = (f > 300) & (f < 3400)
        voice_power = np.mean(Pxx[voice_band])
        
        # 检测音乐频谱特征 (更宽的频率范围)
        music_band = (f > 20) & (f < 15000)
        music_power = np.mean(Pxx[music_band])
        
        # 分类决策
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
        """还原音频内容"""
        if not WAV_AVAILABLE:
            print("  音频保存功能不可用")
            return None
        
        try:
            # 1. 去加重（FM 广播使用 75 μs 去加重）
            tau = self.demod_config['deemphasis_tau']
            sample_rate = self.device_config['sample_rate']
            alpha = 1.0 / (1.0 + 2 * np.pi * sample_rate * tau)
            b = [alpha]
            a = [1, -(1 - alpha)]
            audio = signal.lfilter(b, a, demodulated)
            
            # 2. 低通滤波（限制音频带宽）
            nyquist = sample_rate / 2
            cutoff = self.demod_config['audio_cutoff']
            b, a = signal.butter(5, cutoff / nyquist, btype='low')
            audio = signal.lfilter(b, a, audio)
            
            # 3. 重采样到标准音频采样率
            audio_rate = self.demod_config['audio_sample_rate']
            audio = signal.resample(audio, int(len(audio) * audio_rate / sample_rate))
            
            # 4. 归一化
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                audio = audio / max_abs
            
            # 5. 保存为 WAV 文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"audio_{frequency/1e6:.2f}MHz_{timestamp}.wav")
            wav.write(output_file, audio_rate, (audio * 32767).astype(np.int16))
            
            print(f"  音频已保存到: {output_file}")
            
            return audio
        except Exception as e:
            print(f"  音频还原失败: {e}")
            return None
    
    def save_signal_metadata(self, signal_info):
        """保存信号元数据"""
        if not self.output_config['save_metadata']:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_file = os.path.join(self.output_dir, f"signal_metadata_{timestamp}.json")
            
            # 转换频率为MHz以提高可读性
            for signal in signal_info:
                signal['frequency_mhz'] = signal['frequency'] / 1e6
                signal['bandwidth_mhz'] = signal['bandwidth'] / 1e6
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(signal_info, f, indent=2, ensure_ascii=False)
            
            print(f"  信号元数据已保存到: {metadata_file}")
        except Exception as e:
            print(f"  保存信号元数据失败: {e}")

class USRP_Device:
    """USRP设备类"""
    
    def __init__(self, config):
        """初始化USRP设备"""
        self.config = config['device']
        self.usrp = None
        self.initialize_device()
    
    def initialize_device(self):
        """初始化USRP设备"""
        print("初始化USRP设备...")
        try:
            # 创建USRP设备
            self.usrp = uhd.usrp.MultiUSRP()
            
            # 配置基本参数
            self.sample_rate = self.config['sample_rate']
            self.gain = self.config['gain']
            self.antenna = self.config['antenna']
            
            # 设置采样率
            print(f"设置采样率: {self.sample_rate / 1e6} MHz")
            self.usrp.set_rx_rate(self.sample_rate, 0)
            actual_rate = self.usrp.get_rx_rate(0)
            print(f"实际采样率: {actual_rate / 1e6} MHz")
            
            # 设置增益
            print(f"设置增益: {self.gain} dB")
            self.usrp.set_rx_gain(self.gain, 0)
            actual_gain = self.usrp.get_rx_gain(0)
            print(f"实际增益: {actual_gain} dB")
            
            # 设置天线
            print(f"设置天线: {self.antenna}")
            self.usrp.set_rx_antenna(self.antenna, 0)
            actual_antenna = self.usrp.get_rx_antenna(0)
            print(f"实际天线: {actual_antenna}")
            
            # 打印详细设备信息
            print(f"USRP初始化成功: {self.usrp.get_pp_string()}")
            
        except Exception as e:
            print(f"USRP初始化失败: {e}")
            self.usrp = None
    
    def set_frequency(self, freq):
        """设置频率"""
        if not self.usrp:
            return False
        
        try:
            # 尝试使用不同的API路径
            try:
                # 尝试使用 uhd.libpyuhd.types.tune_request
                if hasattr(uhd, 'libpyuhd') and hasattr(uhd.libpyuhd, 'types') and hasattr(uhd.libpyuhd.types, 'tune_request'):
                    tune_req = uhd.libpyuhd.types.tune_request(freq)
                    self.usrp.set_rx_freq(tune_req, 0)
                # 尝试使用 uhd.types.tune_request
                elif hasattr(uhd, 'types') and hasattr(uhd.types, 'tune_request'):
                    tune_req = uhd.types.tune_request(freq)
                    self.usrp.set_rx_freq(tune_req, 0)
                else:
                    # 尝试直接设置频率
                    self.usrp.set_rx_freq(freq, 0)
            except Exception as e:
                print(f"设置频率失败: {e}")
                return False
            
            return True
        except Exception as e:
            print(f"设置频率失败: {e}")
            return False
    
    def receive_samples(self, num_samples=8192):
        """接收USRP数据"""
        if not self.usrp:
            return np.array([], dtype=np.complex64)
        
        try:
            # 构造stream_args
            stream_args = None
            
            # 优先使用官方推荐的方法
            try:
                if hasattr(uhd.usrp, 'StreamArgs'):
                    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
                    stream_args.channels = [0]
                else:
                    # 尝试字典格式
                    stream_args = {
                        "cpu_format": "fc32",
                        "otw_format": "sc16",
                        "channels": [0]
                    }
            except Exception:
                return np.array([], dtype=np.complex64)
            
            # 创建接收流
            try:
                rx_streamer = self.usrp.get_rx_stream(stream_args)
            except Exception:
                return np.array([], dtype=np.complex64)
            
            # 分配缓冲区
            buffer = np.zeros(num_samples, dtype=np.complex64)
            
            # 创建metadata
            metadata = None
            try:
                # 尝试官方推荐的方法
                from uhd import types
                if hasattr(types, 'RXMetadata'):
                    metadata = types.RXMetadata()
                else:
                    # 尝试其他方法
                    from uhd.libpyuhd import types
                    if hasattr(types, 'rx_metadata_t'):
                        metadata = types.rx_metadata_t()
                    elif hasattr(types, 'rx_metadata'):
                        metadata = types.rx_metadata()
                    else:
                        return np.array([], dtype=np.complex64)
            except Exception:
                return np.array([], dtype=np.complex64)
            
            # 发送流启动命令
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # 尝试使用start模式
                    if hasattr(types.StreamMode, 'start'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start)
                        stream_cmd.stream_now = True
                        stream_cmd.num_samps = num_samples
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    # 回退到start_cont模式
                    elif hasattr(types.StreamMode, 'start_cont'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start_cont)
                        stream_cmd.stream_now = True
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    else:
                        return np.array([], dtype=np.complex64)
                else:
                    # 尝试libpyuhd路径
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 尝试找到启动模式
                        start_mode = None
                        # 先尝试连续模式
                        mode_options = ['STREAM_MODE_START_CONT', 'stream_mode_start_cont', 'START_CONT', 'STREAM_MODE_START_CONTINUOUS', 'start_continuous']
                        for mode_name in mode_options:
                            if hasattr(types.stream_cmd_t, mode_name):
                                start_mode = getattr(types.stream_cmd_t, mode_name)
                                break
                        
                        # 如果找不到连续模式，尝试单次模式
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
                            # 尝试使用数字值直接
                            stream_cmd = types.stream_cmd_t(1)  # 1 is usually START_CONTINUOUS
                            stream_cmd.stream_now = True
                            rx_streamer.issue_stream_cmd(stream_cmd)
                    else:
                        return np.array([], dtype=np.complex64)
            except Exception:
                # 尝试使用连续模式作为最后手段
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 尝试使用数字值直接
                        stream_cmd = types.stream_cmd_t(1)  # 1 is usually START_CONTINUOUS
                        stream_cmd.stream_now = True
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    else:
                        return np.array([], dtype=np.complex64)
                except Exception:
                    return np.array([], dtype=np.complex64)
            
            # 接收数据
            total_received = 0
            max_attempts = 3
            attempts = 0
            
            while attempts < max_attempts and total_received < num_samples:
                try:
                    # 接收数据，使用更长的超时时间
                    num_rx = rx_streamer.recv(buffer[total_received:], metadata, timeout=1.0)
                    
                    # 检查接收状态
                    error_code = getattr(metadata, 'error_code', None)
                    
                    # 处理错误
                    if error_code and error_code != 0:
                        attempts += 1
                        continue
                    
                    # 成功接收
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
            
            # 发送流停止命令
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # 尝试使用stop模式
                    if hasattr(types.StreamMode, 'stop'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop)
                        rx_streamer.issue_stream_cmd(stop_cmd)
                    # 回退到stop_cont模式
                    elif hasattr(types.StreamMode, 'stop_cont'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop_cont)
                        rx_streamer.issue_stream_cmd(stop_cmd)
                else:
                    # 尝试libpyuhd路径
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 尝试找到停止模式
                        stop_mode = None
                        # 先尝试连续模式的停止
                        mode_options = ['STREAM_MODE_STOP_CONT', 'stream_mode_stop_cont', 'STOP_CONT', 'STREAM_MODE_STOP_CONTINUOUS', 'stop_continuous']
                        for mode_name in mode_options:
                            if hasattr(types.stream_cmd_t, mode_name):
                                stop_mode = getattr(types.stream_cmd_t, mode_name)
                                break
                        
                        # 如果找不到连续模式的停止，尝试普通停止
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
                            # 尝试使用数字值直接
                            stop_cmd = types.stream_cmd_t(2)  # 2 is usually STOP_CONTINUOUS
                            rx_streamer.issue_stream_cmd(stop_cmd)
            except Exception:
                # 尝试使用停止模式作为最后手段
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 尝试使用数字值直接
                        stop_cmd = types.stream_cmd_t(2)  # 2 is usually STOP_CONTINUOUS
                        rx_streamer.issue_stream_cmd(stop_cmd)
                except Exception:
                    pass
            
            # 检查结果
            if total_received > 0:
                return buffer[:total_received]
            else:
                return np.array([], dtype=np.complex64)
            
        except Exception:
            return np.array([], dtype=np.complex64)

class SignalVisualizer:
    """信号可视化类"""
    
    def __init__(self, config):
        """初始化可视化器"""
        self.config = config['visualization']
        self.sample_rate = config['device']['sample_rate']
        self.bands = config['scan']['bands']
        
        # 创建图表
        self.fig = plt.figure(figsize=tuple(self.config['figsize']))
        self.fig.suptitle('FM信号处理器 - 实时可视化', fontsize=16)
        
        # 主频谱图
        self.ax_spectrum = self.fig.add_subplot(3, 2, 1)
        self.ax_spectrum.set_title('实时频谱')
        self.ax_spectrum.set_xlabel('频率 (MHz)')
        self.ax_spectrum.set_ylabel('信号强度 (dB)')
        self.ax_spectrum.grid(True)
        
        # 时域图
        self.ax_time = self.fig.add_subplot(3, 2, 3)
        self.ax_time.set_title('时域信号')
        self.ax_time.set_xlabel('时间 (ms)')
        self.ax_time.set_ylabel('幅度')
        self.ax_time.grid(True)
        
        # 扫描进度图
        self.ax_progress = self.fig.add_subplot(3, 2, 4)
        self.ax_progress.set_title('扫描进度')
        self.ax_progress.set_xlabel('频率 (MHz)')
        self.ax_progress.set_ylabel('信号强度 (dB)')
        self.ax_progress.grid(True)
        
        # 信号列表
        self.ax_signals = self.fig.add_subplot(3, 2, 5)
        self.ax_signals.set_title('检测到的信号')
        self.ax_signals.set_xlabel('频率 (MHz)')
        self.ax_signals.set_ylabel('信号强度 (dB)')
        self.ax_signals.grid(True)
        
        # 信号详情
        self.ax_details = self.fig.add_subplot(3, 2, 6)
        self.ax_details.set_title('信号详情')
        self.ax_details.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 初始化数据存储
        self.spectrum_data = []
        self.time_data = []
        self.progress_data = {}
        self.signal_data = []
        for band in self.bands:
            self.progress_data[band['name']] = []
        
        # 初始化图表元素
        self.line_spectrum, = self.ax_spectrum.plot([], [], 'b-')
        self.line_time, = self.ax_time.plot([], [], 'g-')
        self.band_lines = {}
        for band in self.bands:
            line, = self.ax_progress.plot([], [], '-', color=band['color'], label=band['name'])
            self.band_lines[band['name']] = line
        
        # 添加图例
        self.ax_progress.legend()
        
        # 当前状态
        self.current_band_index = 0
        self.current_freq = self.bands[0]['start']
        self.scan_count = 0
        self.detected_signals = []
    
    def update_spectrum(self, samples, current_freq):
        """更新频谱图"""
        # 计算频谱
        spectrum = np.abs(np.fft.fft(samples[:self.config['fft_size']]))
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # 计算频率轴
        freqs = np.fft.fftfreq(self.config['fft_size'], 1/self.sample_rate) / 1e6  # 转换为MHz
        center_freq = current_freq / 1e6
        actual_freqs = freqs + center_freq
        
        # 更新频谱数据
        self.spectrum_data.append(spectrum_db)
        if len(self.spectrum_data) > self.config['window_size']:
            self.spectrum_data.pop(0)
        
        # 更新频谱图
        self.line_spectrum.set_data(actual_freqs, spectrum_db)
        self.ax_spectrum.set_xlim(min(actual_freqs), max(actual_freqs))
        self.ax_spectrum.set_ylim(np.min(spectrum_db) - 5, np.max(spectrum_db) + 5)
        
        # 更新标题
        self.ax_spectrum.set_title(f'实时频谱 - 当前: {center_freq:.2f} MHz')
        
        return [self.line_spectrum]
    
    def update_time(self, samples):
        """更新时域图"""
        # 计算时间轴
        t = np.arange(len(samples[:self.config['fft_size']])) / self.sample_rate * 1000  # 转换为ms
        
        # 使用样本的实部进行时域显示
        time_signal = np.real(samples[:self.config['fft_size']])
        
        # 更新时间数据
        self.time_data.append(time_signal)
        if len(self.time_data) > self.config['window_size']:
            self.time_data.pop(0)
        
        # 更新时域图
        self.line_time.set_data(t, time_signal)
        self.ax_time.set_xlim(0, max(t))
        self.ax_time.set_ylim(np.min(time_signal) - 0.1, np.max(time_signal) + 0.1)
        
        return [self.line_time]
    
    def update_progress(self, band_name, current_freq, signal_strength):
        """更新扫描进度图"""
        # 添加当前数据点
        self.progress_data[band_name].append((current_freq / 1e6, signal_strength))
        
        # 限制数据点数量
        max_points = 1000
        if len(self.progress_data[band_name]) > max_points:
            self.progress_data[band_name] = self.progress_data[band_name][-max_points:]
        
        # 更新所有频段的进度图
        for band in self.bands:
            if self.progress_data[band['name']]:
                freqs, strengths = zip(*self.progress_data[band['name']])
                self.band_lines[band['name']].set_data(freqs, strengths)
                
                # 更新坐标轴范围
                all_freqs = []
                all_strengths = []
                for b in self.bands:
                    if self.progress_data[b['name']]:
                        f, s = zip(*self.progress_data[b['name']])
                        all_freqs.extend(f)
                        all_strengths.extend(s)
                
                if all_freqs:
                    self.ax_progress.set_xlim(min(all_freqs) - 1, max(all_freqs) + 1)
                if all_strengths:
                    self.ax_progress.set_ylim(min(all_strengths) - 5, max(all_strengths) + 5)
        
        return list(self.band_lines.values())
    
    def update_signals(self, signals):
        """更新检测到的信号图"""
        self.detected_signals = signals
        
        # 清空信号图
        self.ax_signals.clear()
        self.ax_signals.set_title('检测到的信号')
        self.ax_signals.set_xlabel('频率 (MHz)')
        self.ax_signals.set_ylabel('信号强度 (dB)')
        self.ax_signals.grid(True)
        
        # 绘制信号
        if signals:
            freqs = [s['frequency'] / 1e6 for s in signals]
            strengths = [s['snr'] for s in signals]
            self.ax_signals.scatter(freqs, strengths, color='red', marker='o')
            
            # 添加信号标签
            for i, (freq, strength) in enumerate(zip(freqs, strengths)):
                self.ax_signals.annotate(f"{freq:.2f}MHz", (freq, strength), fontsize=8)
            
            # 设置坐标轴范围
            self.ax_signals.set_xlim(min(freqs) - 1, max(freqs) + 1)
            self.ax_signals.set_ylim(min(strengths) - 5, max(strengths) + 5)
        
        return [self.ax_signals]
    
    def update_details(self, current_signal=None):
        """更新信号详情"""
        self.ax_details.clear()
        self.ax_details.set_title('信号详情')
        
        if current_signal:
            details = [
                f"频率: {current_signal['frequency']/1e6:.2f} MHz",
                f"带宽: {current_signal['bandwidth']/1e6:.2f} MHz",
                f"信噪比: {current_signal['snr']:.2f} dB",
                f"信号类型: {current_signal.get('type', '未知')}"
            ]
            
            for i, detail in enumerate(details):
                self.ax_details.text(0.1, 0.8 - i*0.15, detail, fontsize=10)
        else:
            self.ax_details.text(0.1, 0.5, "无信号选中", fontsize=12)
        
        return [self.ax_details]
    
    def update_scan(self, scanner, processor):
        """更新扫描状态"""
        # 获取当前频段
        current_band = self.bands[self.current_band_index]
        
        # 设置频率
        scanner.set_frequency(self.current_freq)
        
        # 接收数据
        samples = scanner.receive_samples(num_samples=self.config['fft_size'])
        
        artists = []
        
        if len(samples) == self.config['fft_size']:
            # 计算信号强度
            signal_strength = processor.calculate_snr(samples)
            
            # 估计带宽
            bandwidth = processor.estimate_bandwidth(samples, self.sample_rate)
            
            # 检测信号
            snr_threshold = self.config.get('signal_detection', {}).get('snr_threshold', 10)
            bandwidth_min = self.config.get('signal_detection', {}).get('bandwidth_min', 0.05e6)
            bandwidth_max = self.config.get('signal_detection', {}).get('bandwidth_max', 0.3e6)
            
            is_fm_signal = False
            if bandwidth_min < bandwidth < bandwidth_max and signal_strength > snr_threshold:
                is_fm_signal = True
                # 检查是否已经检测过该信号
                signal_exists = False
                for sig in self.detected_signals:
                    if abs(sig['frequency'] - self.current_freq) < current_band['step']:
                        signal_exists = True
                        break
                
                if not signal_exists:
                    # 解调信号以获取类型
                    demodulated = processor.fm_demod(samples)
                    signal_type = processor.classify_signal(demodulated)
                    
                    # 添加新信号
                    new_signal = {
                        'frequency': self.current_freq,
                        'bandwidth': bandwidth,
                        'snr': signal_strength,
                        'type': signal_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.detected_signals.append(new_signal)
                    print(f"检测到新信号: {self.current_freq/1e6:.2f} MHz, 类型: {signal_type}")
            
            # 更新图表
            spectrum_artists = self.update_spectrum(samples, self.current_freq)
            time_artists = self.update_time(samples)
            progress_artists = self.update_progress(current_band['name'], self.current_freq, signal_strength)
            signal_artists = self.update_signals(self.detected_signals)
            
            artists.extend(spectrum_artists)
            artists.extend(time_artists)
            artists.extend(progress_artists)
            artists.extend(signal_artists)
            
            # 移动到下一个频率
            self.current_freq += current_band['step']
            
            # 检查频段是否完成
            if self.current_freq > current_band['stop']:
                # 移动到下一个频段
                self.current_band_index = (self.current_band_index + 1) % len(self.bands)
                self.current_freq = self.bands[self.current_band_index]['start']
                self.scan_count += 1
                print(f"扫描周期 {self.scan_count} 完成")
        
        return artists

class FM_Industrial_Processor:
    """工业级FM信号处理器"""
    
    def __init__(self):
        """初始化处理器"""
        # 加载配置
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        
        # 初始化设备
        self.device = USRP_Device(self.config)
        
        # 初始化信号处理器
        self.processor = SignalProcessor(self.config)
        
        # 初始化可视化器
        self.visualizer = SignalVisualizer(self.config)
        
    def start_scan(self):
        """开始扫描"""
        print("开始FM信号扫描...")
        
        # 定义动画更新函数
        def update(frame):
            """更新动画"""
            return self.visualizer.update_scan(self.device, self.processor)
        
        # 开始动画
        ani = FuncAnimation(
            self.visualizer.fig,
            update,
            interval=self.config['visualization']['update_interval'],
            blit=True
        )
        
        # 显示图表
        try:
            plt.show()
        except KeyboardInterrupt:
            print("用户中断")
        
        # 保存检测到的信号
        if self.visualizer.detected_signals:
            print(f"\n检测到 {len(self.visualizer.detected_signals)} 个FM信号")
            self.processor.save_signal_metadata(self.visualizer.detected_signals)
            
            # 解调所有检测到的信号
            try:
                demodulate = input("是否解调所有检测到的FM信号? (y/n): ")
                if demodulate.lower() == "y":
                    self.demodulate_all_signals()
            except ValueError:
                pass
        else:
            print("未检测到FM信号")
    
    def demodulate_all_signals(self):
        """解调所有检测到的信号"""
        signals = self.visualizer.detected_signals
        if not signals:
            print("没有检测到FM信号，无法解调")
            return
        
        print("\n" + "="*80)
        print("开始解调所有检测到的FM信号:")
        print("="*80)
        
        for i, signal_info in enumerate(signals):
            frequency = signal_info['frequency']
            print(f"\n解调信号 {i+1}: {frequency/1e6:.2f} MHz")
            
            # 配置频率
            if not self.device.set_frequency(frequency):
                print("  无法设置频率，跳过此信号")
                continue
            
            time.sleep(0.1)
            
            # 接收数据
            samples = self.device.receive_samples(num_samples=32768)  # 更多样本用于解调
            
            if len(samples) == 0:
                print("  未接收到数据，跳过此信号")
                continue
            
            # 解调
            demodulated = self.processor.fm_demod(samples)
            
            # 分类
            signal_type = self.processor.classify_signal(demodulated)
            print(f"  信号类型: {signal_type}")
            
            # 还原音频
            if signal_type in ["VOICE", "MUSIC", "AUDIO"]:
                self.processor.restore_audio(demodulated, frequency)
            else:
                print("  无法识别的信号类型，跳过音频还原")

    def run(self):
        """运行处理器"""
        print("工业级FM信号处理系统")
        print("=" * 60)
        print(f"配置文件: {self.config_manager.config_file}")
        print(f"输出目录: {self.config['output']['output_dir']}")
        print("=" * 60)
        
        if not self.device.usrp:
            print("设备初始化失败，无法继续")
            return
        
        # 开始扫描
        self.start_scan()
        
        print("\n脚本结束")

if __name__ == "__main__":
    processor = FM_Industrial_Processor()
    processor.run()
