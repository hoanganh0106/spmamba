"""Phát trực tiếp (Real-time Streaming) 3 kênh âm thanh sau tách ra Loa / Tai nghe / Bluetooth.

Kịch bản này KHÔNG cần lưu ra file .wav trung gian mà phát thẳng luồng âm thanh PCM
ra thiết bị âm thanh đầu ra mặc định của Laptop (Loa ngoài, Tai nghe 3.5mm, Loa Bluetooth),
hoặc điều khiển bo mạch Kendryte K210 qua cổng USB-UART.

Trong lúc âm thanh đang phát trực tiếp, bấm phím trên bàn phím để chuyển kênh tức thì:
  [Phím 1] -> Chỉ nghe Người nói 1 (Speaker 1)
  [Phím 2] -> Chỉ nghe Người nói 2 (Speaker 2)
  [Phím 3] -> Chỉ nghe Người nói 3 (Speaker 3)
  [Phím 0] -> Nghe âm thanh gốc chưa tách (Original Mixture) để so sánh
  [Phím q] -> Thoát chương trình

Cách dùng:
  1. Phát trực tiếp từ Mic của Laptop (thu Mic -> tách 3 âm -> phát ra Loa/Bluetooth):
     python realtime_stream_pc.py --ckpt best_model.pth --source mic

  2. Phát trực tiếp từ 1 mẫu trong tập data.h5 (hoặc file âm thanh đầu vào) lặp liên tục:
     python realtime_stream_pc.py --ckpt best_model.pth --source h5 --h5_path data/data.h5 --sample_idx 0

  3. Kết nối với bo mạch Kendryte K210 qua cổng USB/Bluetooth Serial (gửi lệnh chọn kênh 1/2/3):
     python realtime_stream_pc.py --source k210_usb --serial_port COM3
"""

import argparse
import os
import sys
import threading
import time
from typing import Optional

from export_k210 import load_student_checkpoint


class KeyboardChannelSelector:
    """Bộ lắng nghe phím bấm thời gian thực (1, 2, 3, 0, q) không làm nghẽn luồng phát âm thanh."""

    def __init__(self, initial_channel: int = 1):
        # 0 = Mixture gốc, 1 = Speaker 1, 2 = Speaker 2, 3 = Speaker 3
        self.active_channel: int = initial_channel
        self.running: bool = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def get_channel(self) -> int:
        with self._lock:
            return self.active_channel

    def set_channel(self, ch: int) -> None:
        with self._lock:
            self.active_channel = ch
        label = "HỖN HỢP GỐC (Mix)" if ch == 0 else f"NGƯỜI NÓI {ch} (Speaker {ch})"
        print(f"\r>>> [ĐANG PHÁT RA LOA]: Kênh {ch} — {label}          ", end="", flush=True)

    def _poll_loop(self) -> None:
        if os.name == "nt":
            import msvcrt

            while self.running:
                if msvcrt.kbhit():
                    ch_bytes = msvcrt.getch()
                    self._handle_key(ch_bytes.decode("utf-8", errors="ignore"))
                time.sleep(0.03)
        else:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            try:
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except Exception:
                old_settings = None
            try:
                while self.running:
                    dr, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if dr:
                        key = sys.stdin.read(1)
                        self._handle_key(key)
            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str) -> None:
        key = key.strip().lower()
        if key in ("0", "1", "2", "3"):
            self.set_channel(int(key))
        elif key == "q":
            print("\n[INFO] Đã nhấn 'q' — Đang dừng phát âm thanh...")
            self.running = False


def stream_from_model_on_pc(
    ckpt_path: str,
    conf_dir: str,
    source: str,
    h5_path: Optional[str] = None,
    sample_idx: int = 0,
    window_sec: float = 1.0,
    initial_channel: int = 1,
) -> None:
    """Chạy K210SepUNet và phát trực tiếp ra Loa/Tai nghe/Bluetooth của Laptop."""
    import numpy as np
    import torch

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Cần cài thư viện phát âm thanh thời gian thực: pip install sounddevice"
        ) from exc

    from look2hear.models.k210_unet import Resampler16kTo8k

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_student_checkpoint(ckpt_path, conf_dir=conf_dir).to(device)
    model.eval()
    model.export_k210 = False

    sr_out = 8000
    chunk_samples = int(window_sec * sr_out)  # Mặc định 1.0s = 8000 mẫu @ 8kHz
    hop_samples = chunk_samples // 2          # Trượt cửa sổ 50% (Overlap-Add mượt không tiếng bụp)

    selector = KeyboardChannelSelector(initial_channel=initial_channel)
    print("\n" + "=" * 68)
    print("BỘ PHÁT TRỰC TIẾP 3 KÊNH ÂM THANH TÁCH (REAL-TIME SPEAKER STREAMING)")
    print("=" * 68)
    print("Hướng dẫn bấm phím trực tiếp:")
    print("  [1] Phát Người nói 1   | [2] Phát Người nói 2   | [3] Phát Người nói 3")
    print("  [0] Phát Âm trộn gốc   | [q] Thoát")
    print("=" * 68)
    selector.start()
    selector.set_channel(initial_channel)

    if source == "h5":
        import h5py

        if not h5_path or not os.path.isfile(h5_path):
            raise FileNotFoundError(f"Không tìm thấy file H5: {h5_path}")
        with h5py.File(h5_path, "r") as hf:
            if "mixture_ids" in hf:
                mid = hf["mixture_ids"][sample_idx]
                if isinstance(mid, bytes):
                    mid = mid.decode("utf-8")
                mix_arr = hf["mixtures"][mid]["mix"][:]
            elif "mixtures" in hf and isinstance(hf["mixtures"], h5py.Group):
                mids = list(hf["mixtures"].keys())
                mix_arr = hf["mixtures"][mids[sample_idx]]["mix"][:]
            else:
                split = "test" if "test" in hf else ("val" if "val" in hf else "train")
                mix_arr = hf[split]["mixtures"][sample_idx]
            mix_16k = torch.from_numpy(mix_arr.astype(np.float32)).unsqueeze(0).to(device)

        resampler = Resampler16kTo8k().to(device)
        with torch.no_grad():
            mix_8k = resampler(mix_16k)                      # [1, T_8k]
            est_8k = model(mix_8k)                           # [1, 3, T_8k]

        mix_np = mix_8k.squeeze(0).cpu().numpy()             # [T_8k]
        est_np = est_8k.squeeze(0).cpu().numpy()             # [3, T_8k]

        # Chuẩn hoá biên độ an toàn trong khoảng [-0.95, 0.95] để bảo vệ loa
        peak = max(float(np.max(np.abs(mix_np))), float(np.max(np.abs(est_np))), 1e-5)
        mix_np = (mix_np / peak) * 0.9
        est_np = (est_np / peak) * 0.9

        total_len = mix_np.shape[0]
        cursor = 0
        block_size = 1024  # 128 ms mỗi lần đẩy ra card âm thanh

        def audio_callback(outdata, frames, time_info, status):
            nonlocal cursor
            ch = selector.get_channel()
            src_buf = mix_np if ch == 0 else est_np[ch - 1]
            out = np.zeros((frames,), dtype=np.float32)
            rem = frames
            write_pos = 0
            while rem > 0:
                avail = total_len - cursor
                take = min(rem, avail)
                out[write_pos : write_pos + take] = src_buf[cursor : cursor + take]
                cursor = (cursor + take) % total_len
                write_pos += take
                rem -= take
            outdata[:, 0] = out

        with sd.OutputStream(
            samplerate=sr_out,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while selector.running:
                time.sleep(0.1)

    elif source == "mic":
        # Thu trực tiếp từ Microphone 8 kHz -> chạy K210SepUNet -> phát ra Loa/Tai nghe
        in_ring = np.zeros((chunk_samples,), dtype=np.float32)
        out_fade_prev = np.zeros((hop_samples,), dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, hop_samples, dtype=np.float32)
        fade_out = 1.0 - fade_in

        with sd.Stream(
            samplerate=sr_out,
            blocksize=hop_samples,
            channels=1,
            dtype="float32",
        ) as stream:
            while selector.running:
                indata, _ = stream.read(hop_samples)
                in_chunk = indata[:, 0]
                in_ring[:hop_samples] = in_ring[hop_samples:]
                in_ring[hop_samples:] = in_chunk

                ch = selector.get_channel()
                if ch == 0:
                    play_chunk = in_chunk
                else:
                    with torch.no_grad():
                        wav_t = torch.from_numpy(in_ring).unsqueeze(0).to(device)
                        est_t = model(wav_t)  # [1, 3, chunk_samples]
                        spk_wav = est_t[0, ch - 1].cpu().numpy()
                    # Cross-fade nửa đầu và lưu nửa sau để khử tiếng khấc giữa các khung
                    play_chunk = spk_wav[:hop_samples] * fade_in + out_fade_prev * fade_out
                    out_fade_prev = spk_wav[hop_samples:].copy()

                stream.write(play_chunk.reshape(-1, 1))


def stream_control_k210_usb(serial_port: str, baudrate: int = 921600, initial_channel: int = 1) -> None:
    """Kết nối Laptop với bo mạch Kendryte K210 qua cổng USB-Serial / Bluetooth UART.

    - Khi bấm 1, 2, 3, 0 trên bàn phím Laptop, lệnh được gửi tức thì xuống K210 để
      mạch K210 phát kênh tương ứng ra loa I2S hoặc truyền ngược luồng PCM 8kHz về Laptop.
    """
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Cần cài pyserial để giao tiếp USB với K210: pip install pyserial") from exc

    selector = KeyboardChannelSelector(initial_channel=initial_channel)
    selector.start()

    print(f"[INFO] Đang mở cổng giao tiếp K210 tại {serial_port} ({baudrate} baud)...")
    with serial.Serial(serial_port, baudrate=baudrate, timeout=0.05) as ser:
        last_ch = -1
        while selector.running:
            ch = selector.get_channel()
            if ch != last_ch:
                ser.write(str(ch).encode("ascii"))
                ser.flush()
                last_ch = ch
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(
        description="Phát trực tiếp (Real-time) 3 kênh âm thanh tách ra Loa/Bluetooth/K210"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["mic", "h5", "k210_usb"],
        default="h5",
        help="Nguồn phát: 'mic' (thu Mic trực tiếp), 'h5' (lấy mẫu từ data.h5), 'k210_usb' (điều khiển bo K210)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="Experiments/checkpoint/K210-SepUNet-Distill-A100/best_model.pth",
        help="Đường dẫn tới checkpoint K210SepUNet đã huấn luyện",
    )
    parser.add_argument(
        "--conf_dir",
        type=str,
        default="configs/distill-k210-a100.yml",
        help="File YAML cấu hình Student",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="data/data.h5",
        help="Đường dẫn tới data.h5 khi dùng --source h5",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Chỉ số mẫu trong data.h5 để phát thử",
    )
    parser.add_argument(
        "--serial_port",
        type=str,
        default="COM3",
        help="Cổng COM/TTY của bo mạch Kendryte K210 khi dùng --source k210_usb",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Kênh phát mặc định lúc khởi động (1, 2, 3 hoặc 0 = Mix gốc)",
    )
    args = parser.parse_args()

    if args.source == "k210_usb":
        stream_control_k210_usb(args.serial_port, initial_channel=args.channel)
    else:
        stream_from_model_on_pc(
            ckpt_path=args.ckpt,
            conf_dir=args.conf_dir,
            source=args.source,
            h5_path=args.h5_path,
            sample_idx=args.sample_idx,
            initial_channel=args.channel,
        )


if __name__ == "__main__":
    main()
