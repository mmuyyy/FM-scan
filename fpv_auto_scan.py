#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPV 5.8G 模拟图传自动扫描脚本
功能：自动扫描 5.725-5.925 GHz 频段，识别图传信号，自动锁定并解调显示
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
# 普通 FM 广播频段 (87.5-108 MHz)
START_FREQ = 87.5e6  # 起始频率 (87.5 MHz)
STOP_FREQ = 108e6    # 结束频率 (108 MHz)
STEP_FREQ = 0.1e6    # 步进频率 (0.1 MHz，适合 FM 广播)
SAMPLE_RATE = 500e3   # 采样率 (500 kHz，降低数据量减少溢出)
GAIN = 30            # 增益 (30 dB，适度降低减少噪声)
BANDWIDTH_LOW = 0.05e6 # 带宽下限 (FM 广播带宽，最小 50 kHz)
BANDWIDTH_HIGH = 0.3e6 # 带宽上限 (FM 广播带宽，最大 300 kHz)
VIDEO_SYNC_RATE = 15625  # 视频行同步频率

# 存储检测到的FM信号
fm_signals = []

class FPV_Auto_Scanner:
    """FPV 自动扫描器类"""
    
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
            
            # 打印设备支持的频段
            try:
                freq_range = self.usrp.get_rx_freq_range(0)
                print(f"设备支持的频率范围: {freq_range.start / 1e6:.2f} - {freq_range.stop / 1e6:.2f} MHz")
            except Exception as e:
                print(f"获取频率范围失败: {e}")
            
            # 打印设备状态
            try:
                sensor_names = self.usrp.get_rx_sensor_names(0)
                print(f"设备传感器: {sensor_names}")
                for sensor in sensor_names:
                    try:
                        value = self.usrp.get_rx_sensor(sensor, 0)
                        print(f"  {sensor}: {value}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"获取传感器信息失败: {e}")
            
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
    
    def scan_and_detect(self, continuous=False):
        """扫频并检测图传信号
        
        Args:
            continuous (bool): 是否持续扫描
        """
        print(f"\n开始扫描 {START_FREQ/1e6} - {STOP_FREQ/1e6} MHz 频段...")
        print(f"扫描模式: {'持续扫描' if continuous else '单次扫描'}")
        
        detected_fpv = None
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
                        
                        # 分析星座图
                        constellation_info = self.analyze_constellation(samples)
                        print(f"  星座图圆度: {constellation_info['circularity']:.3f}")
                        print(f"  相位变化: {constellation_info['phase_variation']:.3f}")
                        
                        # 原始信号检测，无过滤
                        is_fm_signal = False
                        
                        # 带宽判断 - 更宽松的范围
                        bandwidth_match = bandwidth > 0  # 只要有带宽就考虑
                        # 星座图判断
                        constellation_match = constellation_info['is_fm_like']
                        # 信噪比判断 - 更宽松的阈值
                        snr_match = snr > 5  # 非常低的信噪比阈值
                        
                        # 综合判断（非常宽松的条件）
                        if bandwidth_match and constellation_match and snr_match:
                            is_fm_signal = True
                            print(f"  原始检测 - 疑似 FM 信号: 带宽={bandwidth/1e6:.3f} MHz, SNR={snr:.2f} dB, 星座图={constellation_match}")
                            
                            # 进行 FM 解调
                            demodulated = self._fm_demod(samples)
                            
                            # 检测视频同步信号
                            has_sync = self.detect_fm_video_sync(demodulated)
                            
                            # 存储 FM 信号信息
                            fm_signal = {
                                'frequency': current_freq,
                                'bandwidth': bandwidth,
                                'snr': snr,
                                'is_fm_like': constellation_info['is_fm_like'],
                                'circularity': constellation_info['circularity'],
                                'phase_variation': constellation_info['phase_variation'],
                                'mag_stability': constellation_info.get('mag_stability', 0),
                                'power': constellation_info.get('power', 0),
                                'is_fpv': has_sync
                            }
                            fm_signals.append(fm_signal)
                            本轮检测到的信号.append(fm_signal)
                        else:
                            print(f"  原始检测 - 非 FM 信号: 带宽={bandwidth/1e6:.3f} MHz, SNR={snr:.2f} dB, 星座图={constellation_match}")
                    
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
                        signal_type = "FPV 图传" if signal_info['is_fpv'] else "普通 FM"
                        fm_feature = "是" if signal_info.get('is_fm_like', False) else "否"
                        print(f"信号 {i+1}: {signal_info['frequency']/1e6:.2f} MHz, 带宽: {signal_info['bandwidth']/1e6:.2f} MHz, SNR: {signal_info['snr']:.2f} dB")
                        print(f"      星座图: FM 特征={fm_feature}, 圆度={signal_info.get('circularity', 0):.3f}, 相位变化={signal_info.get('phase_variation', 0):.3f}")
                        print(f"      类型: {signal_type}")
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
                signal_type = "FPV 图传" if signal_info['is_fpv'] else "普通 FM"
                fm_feature = "是" if signal_info.get('is_fm_like', False) else "否"
                print(f"信号 {i+1}: {signal_info['frequency']/1e6:.2f} MHz, 带宽: {signal_info['bandwidth']/1e6:.2f} MHz, SNR: {signal_info['snr']:.2f} dB")
                print(f"      星座图: FM 特征={fm_feature}, 圆度={signal_info.get('circularity', 0):.3f}, 相位变化={signal_info.get('phase_variation', 0):.3f}")
                print(f"      类型: {signal_type}")
                print("-" * 80)
        else:
            print("\n扫描完成，未找到 FM 信号")
        
        return detected_fpv
    
    def estimate_bandwidth(self, samples):
        """估计信号带宽"""
        # 计算功率谱密度，使用更大的窗口以获得更准确的结果
        f, Pxx = signal.welch(samples, self.sample_rate, nperseg=1024)
        
        # 找到最大功率
        max_power = np.max(Pxx)
        
        # 找到功率下降 10dB 的频率范围（更适合实际信号）
        threshold = max_power * 0.1  # 10dB 衰减
        
        # 找到所有超过阈值的频率索引
        above_threshold = np.where(Pxx >= threshold)[0]
        
        if len(above_threshold) > 0:
            # 计算带宽 - 考虑正负频率，所以乘以 2
            min_freq = f[above_threshold[0]]
            max_freq = f[above_threshold[-1]]
            # 对于基带信号，带宽是正负频率的总和
            bandwidth = 2 * max_freq
            # 原始数据，无限制
            print(f"  原始带宽测量: {bandwidth/1e6:.3f} MHz")
            return bandwidth
        else:
            print("  未检测到信号，返回 0 带宽")
            return 0
    
    def analyze_constellation(self, samples, num_points=1000):
        """分析星座图，判断调制类型"""
        # 取前 N 个点进行分析
        analysis_samples = samples[:num_points]
        
        # 计算星座点的分布
        i = np.real(analysis_samples)
        q = np.imag(analysis_samples)
        
        # 计算 IQ 点的标准差
        i_std = np.std(i)
        q_std = np.std(q)
        
        # 计算 IQ 点的集中度（圆度）
        # FM 信号的星座图应该是一个圆，IQ 分布均匀
        circularity = abs(i_std - q_std) / max(i_std, q_std, 1e-10)
        
        # 计算信号的相位变化
        phase = np.angle(analysis_samples)
        phase_diff = np.diff(phase)
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        phase_variation = np.std(phase_diff)
        
        # 计算信号的功率
        power = np.mean(np.abs(analysis_samples)**2)
        
        # 计算 IQ 点的归一化幅度
        magnitude = np.abs(analysis_samples)
        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)
        # FM 信号的幅度应该相对稳定
        mag_stability = mag_std / mag_mean
        
        # 原始星座图分析，无过滤
        is_fm_like = False
        # 保持基本判断逻辑，但不过滤结果
        if (circularity < 0.5 and  # 非常宽松的圆度要求
            0.05 < phase_variation < 4.0 and  # 非常宽松的相位变化范围
            mag_stability < 1.0 and  # 非常宽松的幅度稳定性要求
            power > 0.0001):  # 非常低的功率阈值
            is_fm_like = True
        
        # 原始数据，无过滤
        print(f"  原始星座图分析: 圆度={circularity:.3f}, 相位变化={phase_variation:.3f}, 幅度稳定性={mag_stability:.3f}, 功率={power:.3f}, FM特征={is_fm_like}")
        
        return {
            'circularity': circularity,
            'phase_variation': phase_variation,
            'mag_stability': mag_stability,
            'power': power,
            'is_fm_like': is_fm_like
        }
    
    def detect_fm_video_sync(self, demodulated):
        """检测视频行同步信号"""
        # 计算解调信号的功率谱
        f, Pxx = signal.welch(demodulated, self.sample_rate, nperseg=4096)
        
        # 寻找 15625 Hz 的峰值
        sync_freq_idx = np.argmin(np.abs(f - VIDEO_SYNC_RATE))
        
        # 检查该频率点的功率是否显著高于周围频率
        if sync_freq_idx > 0 and sync_freq_idx < len(f) - 1:
            # 计算周围频率的平均功率
            window = 10
            start = max(0, sync_freq_idx - window)
            end = min(len(f), sync_freq_idx + window)
            
            # 排除同步频率点本身
            mask = np.ones(end - start, dtype=bool)
            mask[sync_freq_idx - start] = False
            
            avg_power = np.mean(Pxx[start:end][mask])
            sync_power = Pxx[sync_freq_idx]
            
            # 如果同步频率的功率是周围的 3 倍以上，则认为检测到同步信号
            if sync_power > 3 * avg_power:
                print(f"  检测到视频同步信号: {VIDEO_SYNC_RATE} Hz")
                return True
        
        return False
    
    def start_live_demod(self, freq, bandwidth):
        """锁定频率后实时解调并显示波形"""
        print(f"\n开始实时解调: {freq/1e6:.2f} MHz")
        
        # 配置 USRP 到锁定频率
        if not self._set_frequency(freq):
            return
        time.sleep(0.1)
        
        # 初始化音频输出
        p = None
        stream = None
        if AUDIO_AVAILABLE:
            try:
                p = pyaudio.PyAudio()
                # 音频采样率设为 48kHz
                audio_rate = 48000
                stream = p.open(format=pyaudio.paFloat32,
                               channels=1,
                               rate=audio_rate,
                               output=True)
                print("音频输出已初始化")
            except Exception as e:
                print(f"音频输出初始化失败: {e}")
        
        # 设置 matplotlib 非阻塞模式
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 时域波形
        line1, = ax1.plot([], [], 'b-')
        ax1.set_xlabel('时间 (ms)')
        ax1.set_ylabel('幅度')
        ax1.set_title(f'FPV 基带波形 - {freq/1e6:.2f} MHz')
        ax1.set_ylim(-1, 1)
        ax1.grid(True)
        
        # 频谱
        line2, = ax2.plot([], [], 'r-')
        ax2.set_xlabel('频率 (kHz)')
        ax2.set_ylabel('功率')
        ax2.set_title('信号频谱')
        ax2.grid(True)
        
        try:
            while True:
                # 接收数据（使用修改后的 _receive_samples 方法）
                samples = self._receive_samples()
                
                if len(samples) > 0:
                    # FM 解调
                    demodulated = self._fm_demod(samples)
                    
                    # 生成时间轴
                    t = np.arange(len(demodulated)) / self.sample_rate * 1000  # 转换为毫秒
                    
                    # 更新时域波形
                    line1.set_data(t, demodulated)
                    ax1.set_xlim(t[0], t[-1])
                    
                    # 计算并更新频谱
                    f, Pxx = signal.welch(demodulated, self.sample_rate, nperseg=1024)
                    f_khz = f / 1000  # 转换为 kHz
                    line2.set_data(f_khz, 10 * np.log10(Pxx))
                    ax2.set_xlim(0, self.sample_rate / 2000)  # 显示到 Nyquist 频率
                    ax2.set_ylim(-60, 20)
                    
                    # 音频输出
                    if AUDIO_AVAILABLE and stream and stream.is_active():
                        # 重采样到音频采样率
                        audio_data = signal.resample(demodulated, int(len(demodulated) * 48000 / self.sample_rate))
                        # 转换为 float32 并播放
                        audio_data = audio_data.astype(np.float32)
                        try:
                            stream.write(audio_data.tobytes())
                        except Exception as e:
                            print(f"音频播放出错: {e}")
                    
                    # 保存解调数据（用于后续处理）
                    np.save(f"demodulated_{freq/1e6:.2f}MHz.npy", demodulated)
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
                # 短暂暂停，避免 CPU 占用过高
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n用户中断，停止解调")
        finally:
            # 关闭音频流
            if AUDIO_AVAILABLE and stream:
                stream.stop_stream()
                stream.close()
                if p:
                    p.terminate()
            plt.close(fig)
    
    def _receive_samples(self, num_samples=8192):
        """接收 USRP 数据 - 优化版本，处理 overflow 和 out of sequence 错误"""
        try:
            # 使用更大的缓冲区
            buffer_size = max(num_samples, 8192)
            print(f"\n=== 开始接收数据 ===")
            print(f"目标频率: {self.usrp.get_rx_freq(0) / 1e6:.2f} MHz")
            print(f"缓冲区大小: {buffer_size}")
            
            # 1. 构造 stream_args (简化版)
            print("\n1. 构造 stream_args...")
            stream_args = None
            
            # 优先使用官方推荐的方法
            try:
                if hasattr(uhd.usrp, 'StreamArgs'):
                    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
                    stream_args.channels = [0]
                    print("✓ uhd.usrp.StreamArgs 构造成功")
                else:
                    # 尝试字典格式
                    stream_args = {
                        "cpu_format": "fc32",
                        "otw_format": "sc16",
                        "channels": [0]
                    }
                    print("✓ 字典格式构造成功")
            except Exception as e:
                print(f"  stream_args 构造失败: {str(e)}")
                return np.array([], dtype=np.complex64)
            
            # 2. 创建接收流
            print("\n2. 创建接收流...")
            try:
                rx_streamer = self.usrp.get_rx_stream(stream_args)
                print("✓ 接收流创建成功")
            except Exception as e:
                print(f"✗ 接收流创建失败: {str(e)}")
                return np.array([], dtype=np.complex64)
            
            # 3. 分配缓冲区
            print("\n3. 分配缓冲区...")
            try:
                buffer = np.zeros(buffer_size, dtype=np.complex64)
                print(f"✓ 缓冲区分配成功: {buffer.shape}")
            except Exception as e:
                print(f"✗ 缓冲区分配失败: {str(e)}")
                return np.array([], dtype=np.complex64)
            
            # 4. 创建 metadata
            print("\n4. 创建 metadata...")
            metadata = None
            try:
                # 尝试官方推荐的方法
                from uhd import types
                if hasattr(types, 'RXMetadata'):
                    metadata = types.RXMetadata()
                    print("✓ uhd.types.RXMetadata 创建成功")
                else:
                    # 尝试其他方法
                    from uhd.libpyuhd import types
                    if hasattr(types, 'rx_metadata_t'):
                        metadata = types.rx_metadata_t()
                        print("✓ uhd.libpyuhd.types.rx_metadata_t 创建成功")
                    elif hasattr(types, 'rx_metadata'):
                        metadata = types.rx_metadata()
                        print("✓ uhd.libpyuhd.types.rx_metadata 创建成功")
                    else:
                        print("✗ 无法创建 metadata")
                        return np.array([], dtype=np.complex64)
            except Exception as e:
                print(f"  metadata 创建失败: {str(e)}")
                return np.array([], dtype=np.complex64)
            
            # 5. 发送流启动命令
            print("\n5. 发送流启动命令...")
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # 检查 StreamMode 有哪些属性
                    stream_mode_attrs = [attr for attr in dir(types.StreamMode) if not attr.startswith('__')]
                    print(f"  StreamMode 可用属性: {stream_mode_attrs}")
                    
                    # 尝试使用 start 模式
                    if hasattr(types.StreamMode, 'start'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start)
                        stream_cmd.stream_now = True
                        stream_cmd.num_samps = buffer_size
                        rx_streamer.issue_stream_cmd(stream_cmd)
                        print("✓ 流启动命令发送成功 (使用 start 模式)")
                    # 回退到 start_cont 模式
                    elif hasattr(types.StreamMode, 'start_cont'):
                        stream_cmd = types.StreamCMD(types.StreamMode.start_cont)
                        stream_cmd.stream_now = True
                        rx_streamer.issue_stream_cmd(stream_cmd)
                        print("✓ 流启动命令发送成功 (使用 start_cont 模式)")
                    else:
                        print(f"✗ 找不到合适的 StreamMode，可用属性: {stream_mode_attrs}")
                        return np.array([], dtype=np.complex64)
                else:
                    # 尝试 libpyuhd 路径
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 检查 stream_cmd_t 有哪些属性
                        stream_cmd_attrs = [attr for attr in dir(types.stream_cmd_t) if not attr.startswith('__')]
                        print(f"  stream_cmd_t 可用属性: {stream_cmd_attrs}")
                        
                        # 尝试找到启动模式
                        start_mode = None
                        # 先尝试连续模式（更可靠）
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
                                stream_cmd.num_samps = buffer_size
                            rx_streamer.issue_stream_cmd(stream_cmd)
                            print("✓ 流启动命令发送成功")
                        else:
                            print(f"✗ 无法找到启动模式，可用属性: {stream_cmd_attrs}")
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
                        print("✓ 流启动命令发送成功 (使用数字值 1)")
                    else:
                        return np.array([], dtype=np.complex64)
                except Exception as e2:
                    print(f"  最后的尝试也失败: {str(e2)}")
                    return np.array([], dtype=np.complex64)
            
            # 6. 接收数据 - 带溢出处理
            print("\n6. 接收数据...")
            total_received = 0
            max_attempts = 3
            attempts = 0
            
            while attempts < max_attempts and total_received < buffer_size:
                try:
                    print(f"  尝试接收数据 (尝试 {attempts+1}/{max_attempts})...")
                    # 接收数据，使用更长的超时时间
                    num_rx = rx_streamer.recv(buffer[total_received:], metadata, timeout=1.0)
                    
                    # 检查接收状态
                    error_code = getattr(metadata, 'error_code', None)
                    
                    # 处理错误
                    if error_code and error_code != 0:
                        print(f"  接收错误: {error_code}")
                        
                        # 处理 overflow 错误
                        if 'overflow' in str(error_code).lower():
                            print("  检测到溢出，重置流...")
                            # 发送停止命令
                            try:
                                from uhd import types
                                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                                    stop_cmd = types.StreamCMD(types.StreamMode.stop)
                                    rx_streamer.issue_stream_cmd(stop_cmd)
                            except:
                                pass
                            # 重新发送启动命令
                            try:
                                from uhd import types
                                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                                    stream_cmd = types.StreamCMD(types.StreamMode.start)
                                    stream_cmd.stream_now = True
                                    stream_cmd.num_samps = buffer_size - total_received
                                    rx_streamer.issue_stream_cmd(stream_cmd)
                            except:
                                pass
                            attempts += 1
                            continue
                        
                        # 处理 out of sequence 错误
                        elif 'sequence' in str(error_code).lower():
                            print("  检测到乱序错误，继续接收...")
                            attempts += 1
                            continue
                        
                        # 其他错误
                        else:
                            print(f"  未知错误: {error_code}")
                            break
                    
                    # 成功接收
                    if num_rx > 0:
                        total_received += num_rx
                        print(f"  接收到 {num_rx} 样本，总计 {total_received}")
                        if total_received >= buffer_size:
                            break
                    else:
                        print("  未接收到数据")
                        attempts += 1
                        time.sleep(0.2)  # 增加等待时间
                        
                except Exception as e:
                    print(f"  接收数据异常: {str(e)}")
                    attempts += 1
                    time.sleep(0.2)
            
            # 7. 发送流停止命令
            print("\n7. 发送流停止命令...")
            try:
                from uhd import types
                if hasattr(types, 'StreamCMD') and hasattr(types, 'StreamMode'):
                    # 检查 StreamMode 有哪些属性
                    stream_mode_attrs = [attr for attr in dir(types.StreamMode) if not attr.startswith('__')]
                    print(f"  StreamMode 可用属性: {stream_mode_attrs}")
                    
                    # 尝试使用 stop 模式
                    if hasattr(types.StreamMode, 'stop'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop)
                        rx_streamer.issue_stream_cmd(stop_cmd)
                        print("✓ 流停止命令发送成功 (使用 stop 模式)")
                    # 回退到 stop_cont 模式
                    elif hasattr(types.StreamMode, 'stop_cont'):
                        stop_cmd = types.StreamCMD(types.StreamMode.stop_cont)
                        rx_streamer.issue_stream_cmd(stop_cmd)
                        print("✓ 流停止命令发送成功 (使用 stop_cont 模式)")
                    else:
                        print(f"✗ 找不到合适的 StreamMode，可用属性: {stream_mode_attrs}")
                else:
                    # 尝试 libpyuhd 路径
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 检查 stream_cmd_t 有哪些属性
                        stream_cmd_attrs = [attr for attr in dir(types.stream_cmd_t) if not attr.startswith('__')]
                        print(f"  stream_cmd_t 可用属性: {stream_cmd_attrs}")
                        
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
                            print("✓ 流停止命令发送成功")
                        else:
                            print(f"✗ 无法找到停止模式，可用属性: {stream_cmd_attrs}")
            except Exception as e:
                print(f"  流停止命令发送失败: {str(e)}")
                # 尝试直接使用停止模式作为最后的尝试
                try:
                    from uhd.libpyuhd import types
                    if hasattr(types, 'stream_cmd_t'):
                        # 尝试直接使用数字值
                        stop_cmd = types.stream_cmd_t(2)  # 2 通常是 STOP_CONTINUOUS
                        rx_streamer.issue_stream_cmd(stop_cmd)
                        print("✓ 流停止命令发送成功 (使用数字值 2)")
                except Exception as e2:
                    print(f"  最后的尝试也失败: {str(e2)}")
            
            # 8. 检查结果
            if total_received > 0:
                print(f"\n✓ 数据接收完成，总计 {total_received} 样本")
                return buffer[:total_received]
            else:
                print("\n✗ 未接收到有效数据")
                return np.array([], dtype=np.complex64)
            
        except Exception as e:
            print(f"✗ 接收数据失败: {str(e)}")
            return np.array([], dtype=np.complex64)
    
    def _fm_demod(self, samples):
        """FM 解调
        使用复数信号进行解调，避免类型不匹配问题"""
        # 计算相位差
        phase = np.angle(samples)
        phase_diff = np.diff(phase)
        
        # 处理相位跳变
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        
        # 归一化输出
        demodulated = phase_diff / np.max(np.abs(phase_diff))
        
        return demodulated
    
    def _calculate_snr(self, samples):
        """计算信噪比 - 原始测量，无过滤"""
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
        
        # 计算频谱平坦度（仅用于参考，不用于过滤）
        spectrum_std = np.std(Pxx)
        spectrum_mean = np.mean(Pxx)
        flatness = spectrum_std / spectrum_mean
        
        # 原始数据，无过滤
        print(f"  原始测量 - 信号功率: {signal_power:.6f}, 噪声功率: {noise_power:.6f}, SNR: {snr:.2f} dB, 平坦度: {flatness:.3f}")
        return snr

def main():
    """主函数"""
    print("FPV 5.8G 模拟图传自动扫描脚本")
    print("=" * 60)
    
    # 创建扫描器实例
    scanner = FPV_Auto_Scanner()
    
    # 让用户选择扫描模式
    try:
        scan_mode = input("请选择扫描模式 (1: 单次扫描, 2: 持续扫描): ")
        continuous = (scan_mode == "2")
    except ValueError:
        continuous = False
    
    # 开始扫描
    result = scanner.scan_and_detect(continuous=continuous)
    
    # 处理检测结果
    if fm_signals:
        # 找出所有 FPV 图传信号
        fpv_signals = [signal for signal in fm_signals if signal['is_fpv']]
        
        if fpv_signals:
            print("\n" + "="*60)
            print("找到的 FPV 图传信号:")
            print("="*60)
            for i, signal_info in enumerate(fpv_signals):
                print(f"{i+1}. {signal_info['frequency']/1e6:.2f} MHz, 带宽: {signal_info['bandwidth']/1e6:.2f} MHz, SNR: {signal_info['snr']:.2f} dB")
            
            # 让用户选择要解调的信号
            try:
                choice = int(input("\n请选择要解调的信号编号 (输入 0 退出): "))
                if 1 <= choice <= len(fpv_signals):
                    selected_signal = fpv_signals[choice-1]
                    scanner.start_live_demod(selected_signal['frequency'], selected_signal['bandwidth'])
                else:
                    print("退出解调")
            except ValueError:
                print("无效输入，退出解调")
        else:
            print("\n未找到 FPV 图传信号，但检测到普通 FM 信号")
    
    print("\n脚本结束")

# 代码架构说明
"""
代码架构说明：

1. 核心类: FPV_Auto_Scanner
   - __init__(): 初始化 USRP 设备，配置基本参数
   - scan_and_detect(): 自动扫描频段，检测 FM 信号和 FPV 图传信号
   - estimate_bandwidth(): 估计信号带宽
   - detect_fm_video_sync(): 检测视频行同步信号
   - start_live_demod(): 实时解调并显示波形，支持音频输出
   - _receive_samples(): 接收 USRP 数据
   - _fm_demod(): FM 解调（使用复数信号）
   - _calculate_snr(): 计算信噪比

2. 数据流处理:
   USRP 原始 IQ 数据 (complex float32)
   → 带宽估计
   → FM 解调 (complex → float)
   → 视频同步检测
   → 实时波形显示
   → 音频输出（可选）

3. 扩展功能:
   - 扩展了扫描频段 (5700-6000 MHz)
   - 提高了扫描精度 (1 MHz 步进)
   - 存储所有检测到的 FM 信号
   - 支持音频输出
   - 实时显示时域波形和频谱
   - 保存解调数据用于后续处理

4. 技术特点:
   - 使用复数信号进行 FM 解调，避免类型不匹配问题
   - 非阻塞式实时显示
   - 完整的异常处理
   - 可配置的参数
   - 兼容 Linux 和 Windows 平台
"""

if __name__ == "__main__":
    main()
