#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FM 信号扫描与解调脚本 (第一阶段)
功能：扫描 FM 信号，使用相位差分法解调，基本频谱分类，音频还原
设备：USRP B210 / B200mini
技术栈：Python 3.8-3.11 + uhd + numpy + scipy + matplotlib
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
import sys
import uhd

# 尝试导入 pyaudio 用于音频输出
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("pyaudio 未安装，音频输出功能将不可用")

# 可配置参数
# 扫描参数
START_FREQ = 87.5e6      # 起始频率 (87.5 MHz)
STOP_FREQ = 108e6        # 结束频率 (108 MHz)
STEP_FREQ = 0.1e6        # 步进频率 (0.1 MHz)
SAMPLE_RATE = 2e6        # 采样率 (2 MHz)
GAIN = 30                # 增益 (30 dB)

# 存储检测到的FM信号
fm_signals = []

class FM_Scan_Demod:
    """FM 信号扫描与解调类"""
    
    def __init__(self):
        """初始化 USRP 设备和参数"""
        print("正在初始化 USRP 设备...")
        try:
            # 创建 USRP 设备
            self.usrp = uhd.usrp.MultiUSRP()
            
            # 配置基本参数
            self.sample_rate = SAMPLE_RATE
            self.gain = GAIN
            self.antenna = "RX2"
            
            # 先设置采样率
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
            print(f"USRP 初始化成功: {self.usrp.get_pp_string()}")
            
        except Exception as e:
            print(f"USRP 初始化失败: {e}")
            sys.exit(1)
    
    def _set_frequency(self, freq):
        """设置频率，处理不同版本的 API"""
        try:
            print(f"设置频率: {freq / 1e6:.2f} MHz")
            
            # 尝试使用不同的 API 路径
            try:
                # 尝试使用 uhd.libpyuhd.types.tune_request
                if hasattr(uhd, 'libpyuhd') and hasattr(uhd.libpyuhd, 'types') and hasattr(uhd.libpyuhd.types, 'tune_request'):
                    tune_req = uhd.libpyuhd.types.tune_request(freq)
                    self.usrp.set_rx_freq(tune_req, 0)
                    print("  使用 uhd.libpyuhd.types.tune_request")
                # 尝试使用 uhd.types.tune_request
                elif hasattr(uhd, 'types') and hasattr(uhd.types, 'tune_request'):
                    tune_req = uhd.types.tune_request(freq)
                    self.usrp.set_rx_freq(tune_req, 0)
                    print("  使用 uhd.types.tune_request")
                else:
                    # 尝试直接设置频率
                    self.usrp.set_rx_freq(freq, 0)
                    print("  直接设置频率")
            except Exception as e:
                print(f"  设置频率失败: {e}")
                return False
            
            # 验证频率设置
            actual_freq = self.usrp.get_rx_freq(0)
            print(f"  实际频率: {actual_freq / 1e6:.2f} MHz")
            
            return True
        except Exception as e:
            print(f"设置频率失败: {e}")
            return False
    
    def scan(self, continuous=False):
        """扫描 FM 信号"""
        print(f"\n开始扫描 {START_FREQ/1e6} - {STOP_FREQ/1e6} MHz 频段...")
        print(f"扫描模式: {'持续扫描' if continuous else '单次扫描'}")
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                print(f"\n=== 第 {scan_count} 轮扫描 ===")
                
                current_freq = START_FREQ
                本轮检测到的信号 = []
                
                while current_freq <= STOP_FREQ:
                    print(f"扫描频率: {current_freq/1e6:.2f} MHz")
                    
                    try:
                        # 配置当前频率
                        if not self._set_frequency(current_freq):
                            current_freq += STEP_FREQ
                            continue
                        
                        time.sleep(0.1)  # 频率切换后等待稳定
                        
                        # 接收数据
                        samples = self._receive_samples()
                        
                        # 如果没有接收到数据，直接跳到下一个频率
                        if len(samples) == 0:
                            print(f"  未接收到数据，跳到下一个频率")
                            current_freq += STEP_FREQ
                            continue
                        
                        # 估计带宽
                        bandwidth = self.estimate_bandwidth(samples)
                        print(f"  估计带宽: {bandwidth/1e6:.2f} MHz")
                        
                        # 计算信噪比
                        snr = self._calculate_snr(samples)
                        print(f"  信噪比: {snr:.2f} dB")
                        
                        # 基本信号检测
                        is_fm_signal = False
                        
                        # 带宽判断
                        bandwidth_match = 0.05e6 < bandwidth < 0.3e6  # FM 广播带宽
                        # 信噪比判断
                        snr_match = snr > 10  # 信噪比阈值
                        
                        # 综合判断
                        if bandwidth_match and snr_match:
                            is_fm_signal = True
                            print(f"  检测到 FM 信号: 带宽={bandwidth/1e6:.3f} MHz, SNR={snr:.2f} dB")
                            
                            # 存储 FM 信号信息
                            fm_signal = {
                                'frequency': current_freq,
                                'bandwidth': bandwidth,
                                'snr': snr,
                            }
                            fm_signals.append(fm_signal)
                            本轮检测到的信号.append(fm_signal)
                        else:
                            print(f"  非 FM 信号: 带宽={bandwidth/1e6:.3f} MHz, SNR={snr:.2f} dB")
                    
                    except Exception as e:
                        print(f"  扫描出错: {e}")
                    
                    # 继续下一个频点
                    current_freq += STEP_FREQ
                
                # 打印本轮检测到的信号
                if 本轮检测到的信号:
                    print("\n" + "="*80)
                    print(f"第 {scan_count} 轮扫描结果:")
                    print("="*80)
                    for i, signal_info in enumerate(本轮检测到的信号):
                        print(f"信号 {i+1}: {signal_info['frequency']/1e6:.2f} MHz, 带宽: {signal_info['bandwidth']/1e6:.2f} MHz, SNR: {signal_info['snr']:.2f} dB")
                        print("-" * 80)
                else:
                    print("\n本轮扫描未找到 FM 信号")
                
                # 如果不是持续扫描，则退出
                if not continuous:
                    break
                
                # 持续扫描间隔
                print("\n等待 5 秒后开始下一轮扫描...")
                time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n用户中断扫描")
        
        # 打印所有检测到的 FM 信号
        if fm_signals:
            print("\n" + "="*80)
            print("所有检测到的 FM 信号:")
            print("="*80)
            for i, signal_info in enumerate(fm_signals):
                print(f"信号 {i+1}: {signal_info['frequency']/1e6:.2f} MHz, 带宽: {signal_info['bandwidth']/1e6:.2f} MHz, SNR: {signal_info['snr']:.2f} dB")
                print("-" * 80)
        else:
            print("\n扫描完成，未找到 FM 信号")
    
    def estimate_bandwidth(self, samples):
        """估计信号带宽"""
        # 计算功率谱密度
        f, Pxx = signal.welch(samples, self.sample_rate, nperseg=1024)
        
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
    
    def _calculate_snr(self, samples):
        """计算信噪比"""
        # 计算信号功率
        signal_power = np.mean(np.abs(samples)**2)
        
        # 假设噪声是信号的一部分，通过频谱分析估计噪声
        f, Pxx = signal.welch(samples, self.sample_rate, nperseg=1024)
        
        # 找到信号频段外的噪声
        noise_bands = np.concatenate([Pxx[:50], Pxx[-50:]])
        noise_power = np.mean(noise_bands)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0
        
        return snr
    
    def _receive_samples(self, num_samples=8192):
        """接收 USRP 数据"""
        try:
            # 构造 stream_args
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
            except Exception as e:
                print(f"  stream_args 构造失败: {str(e)}")
                return np.array([], dtype=np.complex64)
            
            # 创建接收流
            try:
                rx_streamer = self.usrp.get_rx_stream(stream_args)
            except Exception as e:
                print(f"  接收流创建失败: {str(e)}")
                return np.array([], dtype=np.complex64)
            
            # 分配缓冲区
            buffer = np.zeros(num_samples, dtype=np.complex64)
            
            # 创建 metadata
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
            except Exception as e:
                print(f"  metadata 创建失败: {str(e)}")
                return np.array([], dtype=np.complex64)
            
            # 发送流启动命令
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # 尝试使用 start 模式
                    if hasattr(types.StreamMode, 'start'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start)
                        stream_cmd.stream_now = True
                        stream_cmd.num_samps = num_samples
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    # 回退到 start_cont 模式
                    elif hasattr(types.StreamMode, 'start_cont'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start_cont)
                        stream_cmd.stream_now = True
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    else:
                        return np.array([], dtype=np.complex64)
                else:
                    # 尝试 libpyuhd 路径
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
                            return np.array([], dtype=np.complex64)
            except Exception as e:
                print(f"  流启动命令发送失败: {str(e)}")
                # 尝试直接使用连续模式作为最后的尝试
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 尝试直接使用数字值
                        stream_cmd = types.stream_cmd_t(1)  # 1 通常是 START_CONTINUOUS
                        stream_cmd.stream_now = True
                        rx_streamer.issue_stream_cmd(stream_cmd)
                    else:
                        return np.array([], dtype=np.complex64)
                except Exception as e2:
                    print(f"  最后的尝试也失败: {str(e2)}")
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
                        print(f"  接收错误: {error_code}")
                        attempts += 1
                        continue
                    
                    # 成功接收
                    if num_rx > 0:
                        total_received += num_rx
                        if total_received >= num_samples:
                            break
                    else:
                        print("  未接收到数据")
                        attempts += 1
                        time.sleep(0.2)
                        
                except Exception as e:
                    print(f"  接收数据异常: {str(e)}")
                    attempts += 1
                    time.sleep(0.2)
            
            # 发送流停止命令
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # 尝试使用 stop 模式
                    if hasattr(types.StreamMode, 'stop'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop)
                        rx_streamer.issue_stream_cmd(stop_cmd)
                    # 回退到 stop_cont 模式
                    elif hasattr(types.StreamMode, 'stop_cont'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop_cont)
                        rx_streamer.issue_stream_cmd(stop_cmd)
                else:
                    # 尝试 libpyuhd 路径
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
            except Exception as e:
                print(f"  流停止命令发送失败: {str(e)}")
                # 尝试直接使用停止模式作为最后的尝试
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 尝试直接使用数字值
                        stop_cmd = types.stream_cmd_t(2)  # 2 通常是 STOP_CONTINUOUS
                        rx_streamer.issue_stream_cmd(stop_cmd)
                except Exception as e2:
                    pass
            
            # 检查结果
            if total_received > 0:
                return buffer[:total_received]
            else:
                return np.array([], dtype=np.complex64)
            
        except Exception as e:
            print(f"  接收数据失败: {str(e)}")
            return np.array([], dtype=np.complex64)
    
    def fm_demod(self, samples):
        """相位差分法 FM 解调"""
        # 计算相位差
        phase = np.angle(samples)
        phase_diff = np.diff(phase)
        
        # 处理相位跳变
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        
        # 归一化输出
        demodulated = phase_diff / np.max(np.abs(phase_diff))
        
        return demodulated
    
    def classify_signal(self, demodulated):
        """基本频谱分类"""
        # 计算功率谱
        f, Pxx = signal.welch(demodulated, self.sample_rate, nperseg=4096)
        
        # 检测音频频谱特征
        audio_band = (f > 20) & (f < 20000)
        audio_power = np.mean(Pxx[audio_band])
        
        # 检测语音频谱特征 (300-3400 Hz)
        voice_band = (f > 300) & (f < 3400)
        voice_power = np.mean(Pxx[voice_band])
        
        # 分类决策
        if audio_power > 0.1 * np.max(Pxx):
            if voice_power > 0.5 * audio_power:
                return "VOICE"
            else:
                return "MUSIC"
        else:
            return "UNKNOWN"
    
    def restore_audio(self, demodulated, frequency):
        """还原音频内容"""
        # 1. 去加重（FM 广播使用 75 μs 去加重）
        tau = 75e-6  # 去加重时间常数
        alpha = 1.0 / (1.0 + 2 * np.pi * self.sample_rate * tau)
        b = [alpha]
        a = [1, -(1 - alpha)]
        audio = signal.lfilter(b, a, demodulated)
        
        # 2. 低通滤波（限制音频带宽）
        nyquist = self.sample_rate / 2
        cutoff = 15000  # 15 kHz
        b, a = signal.butter(5, cutoff / nyquist, btype='low')
        audio = signal.lfilter(b, a, audio)
        
        # 3. 重采样到标准音频采样率
        audio_rate = 48000
        audio = signal.resample(audio, int(len(audio) * audio_rate / self.sample_rate))
        
        # 4. 归一化
        audio = audio / np.max(np.abs(audio))
        
        # 5. 保存为 WAV 文件
        import scipy.io.wavfile as wav
        output_file = f"audio_{frequency/1e6:.2f}MHz.wav"
        wav.write(output_file, audio_rate, (audio * 32767).astype(np.int16))
        
        print(f"  音频已保存到: {output_file}")
        
        return audio
    
    def demodulate_all_signals(self):
        """解调所有检测到的信号"""
        if not fm_signals:
            print("没有检测到 FM 信号，无法解调")
            return
        
        print("\n" + "="*80)
        print("开始解调所有检测到的 FM 信号:")
        print("="*80)
        
        for i, signal_info in enumerate(fm_signals):
            frequency = signal_info['frequency']
            print(f"\n解调信号 {i+1}: {frequency/1e6:.2f} MHz")
            
            # 配置频率
            if not self._set_frequency(frequency):
                print("  无法设置频率，跳过此信号")
                continue
            
            time.sleep(0.1)
            
            # 接收数据
            samples = self._receive_samples(num_samples=32768)  # 更多样本用于解调
            
            if len(samples) == 0:
                print("  未接收到数据，跳过此信号")
                continue
            
            # 解调
            demodulated = self.fm_demod(samples)
            
            # 分类
            signal_type = self.classify_signal(demodulated)
            print(f"  信号类型: {signal_type}")
            
            # 还原音频
            if signal_type in ["VOICE", "MUSIC"]:
                self.restore_audio(demodulated, frequency)
            else:
                print("  无法识别的信号类型，跳过音频还原")


def main():
    """主函数"""
    print("FM 信号扫描与解调脚本")
    print("=" * 60)
    
    # 创建扫描器实例
    scanner = FM_Scan_Demod()
    
    # 让用户选择扫描模式
    try:
        scan_mode = input("请选择扫描模式 (1: 单次扫描, 2: 持续扫描): ")
        continuous = (scan_mode == "2")
    except ValueError:
        continuous = False
    
    # 开始扫描
    scanner.scan(continuous=continuous)
    
    # 解调所有检测到的信号
    if fm_signals:
        try:
            demodulate = input("是否解调所有检测到的 FM 信号? (y/n): ")
            if demodulate.lower() == "y":
                scanner.demodulate_all_signals()
        except ValueError:
            pass
    
    print("\n脚本结束")


if __name__ == "__main__":
    main()
