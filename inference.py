import os
import torch
import soundfile as sf
import look2hear.models
from look2hear.utils import new_complex_like

def separate(model, audio_path, device="cuda"):
    print(f"Bắt đầu tách file {audio_path}...")
    wav, sr = sf.read(audio_path)
    
    # Ensure Mono
    if len(wav.shape) > 1:
        wav = wav[:, 0]
        
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        est_sources = model(wav_tensor)
        
    # Shape is supposed to be (1, n_srcs, time)
    est_sources_np = est_sources.squeeze(0).cpu().numpy()
    
    n_srcs = est_sources_np.shape[0]
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    for i in range(n_srcs):
        out_path = f"{base_name}_spk{i}.wav"
        sf.write(out_path, est_sources_np[i], sr)
        print(f"Đã lưu: {out_path}")
        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "epoch=27-best.ckpt"
    print(f"Đang tải model từ {ckpt_path} lên {device}...")
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    
    config = ckpt["training_config"]
    if "audionet" not in config:
        # Check if the checkpoint has a different structure
        print("Không tìm thấy cấu hình audionet trong ckpt, thử lại.")
        raise ValueError("Missing audionet config")
        
    audionet_name = config["audionet"]["audionet_name"]
    sample_rate = config["datamodule"]["data_config"]["sample_rate"]
    
    print(f"Khởi tạo model {audionet_name} với sample_rate={sample_rate}...")
    model = getattr(look2hear.models, audionet_name)(
        sample_rate=sample_rate,
        **config["audionet"]["audionet_config"]
    )
    
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("audio_model."):
            new_state_dict[k.replace("audio_model.", "")] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Tải model thành công!")
    
    files_to_process = ["mix_16k.wav", "sample_3mix_with_noise.wav"]
    for f in files_to_process:
        if os.path.exists(f):
            separate(model, f, device)
        else:
            print(f"Không tìm thấy file: {f}")
    
    print("Hoàn tất!")