import os
import tempfile
import torchaudio
import uuid
import sys
import shutil
from collections.abc import Mapping

# Function to find ComfyUI directories
def get_comfyui_temp_dir():
    """Dynamically find the ComfyUI temp directory"""
    # First check using folder_paths if available
    try:
        import folder_paths
        comfy_dir = os.path.dirname(os.path.dirname(os.path.abspath(folder_paths.__file__)))
        temp_dir = os.path.join(comfy_dir, "temp")
        return temp_dir
    except:
        pass
    
    # Try to locate based on current script location
    try:
        # This script is likely in a ComfyUI custom nodes directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up until we find the ComfyUI directory
        potential_dir = current_dir
        for _ in range(5):  # Limit to 5 levels up
            if os.path.exists(os.path.join(potential_dir, "comfy.py")):
                return os.path.join(potential_dir, "temp")
            potential_dir = os.path.dirname(potential_dir)
    except:
        pass
    
    # Return None if we can't find it
    return None

def optimize_memory_usage():
    """Optimize memory usage by clearing caches and setting appropriate flags"""
    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Set memory efficient attention
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction to avoid using all GPU memory
        try:
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
        except:
            pass
            
        # Enable memory efficient attention
        try:
            from xformers.ops import memory_efficient_attention
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except:
            pass

def process_frames_memory_efficient(frames, device):
    """Process frames in a memory efficient way"""
    # Process frames in smaller batches to avoid OOM
    batch_size = 4  # Adjust based on available memory
    processed_frames = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        # Convert to bytes while still on CPU
        batch = (batch * 255).byte()
        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)
        processed_frames.append(batch)
        
        # Clear memory after each batch
        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all processed frames
    return torch.cat(processed_frames)

# Function to clean up any ComfyUI temp directories
def cleanup_comfyui_temp_directories():
    """Find and clean up any ComfyUI temp directories"""
    comfyui_temp = get_comfyui_temp_dir()
    if not comfyui_temp:
        print("Could not locate ComfyUI temp directory")
        return
    
    comfyui_base = os.path.dirname(comfyui_temp)
    
    # Check for the main temp directory
    if os.path.exists(comfyui_temp):
        try:
            shutil.rmtree(comfyui_temp)
            print(f"Removed ComfyUI temp directory: {comfyui_temp}")
        except Exception as e:
            print(f"Could not remove {comfyui_temp}: {str(e)}")
            # If we can't remove it, try to rename it
            try:
                backup_name = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
                os.rename(comfyui_temp, backup_name)
                print(f"Renamed {comfyui_temp} to {backup_name}")
            except:
                pass
    
    # Find and clean up any backup temp directories
    try:
        all_directories = [d for d in os.listdir(comfyui_base) if os.path.isdir(os.path.join(comfyui_base, d))]
        for dirname in all_directories:
            if dirname.startswith("temp_backup_"):
                backup_path = os.path.join(comfyui_base, dirname)
                try:
                    shutil.rmtree(backup_path)
                    print(f"Removed backup temp directory: {backup_path}")
                except Exception as e:
                    print(f"Could not remove backup dir {backup_path}: {str(e)}")
    except Exception as e:
        print(f"Error cleaning up temp directories: {str(e)}")

# Create a module-level function to set up system-wide temp directory
def init_temp_directories():
    """Initialize global temporary directory settings"""
    # First clean up any existing temp directories
    cleanup_comfyui_temp_directories()
    
    # Generate a unique base directory for this module
    system_temp = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())[:8]
    temp_base_path = os.path.join(system_temp, f"latentsync_{unique_id}")
    os.makedirs(temp_base_path, exist_ok=True)
    
    # Override environment variables that control temp directories
    os.environ['TMPDIR'] = temp_base_path
    os.environ['TEMP'] = temp_base_path
    os.environ['TMP'] = temp_base_path
    
    # Force Python's tempfile module to use our directory
    tempfile.tempdir = temp_base_path
    
    # Final check for ComfyUI temp directory
    comfyui_temp = get_comfyui_temp_dir()
    if comfyui_temp and os.path.exists(comfyui_temp):
        try:
            shutil.rmtree(comfyui_temp)
            print(f"Removed ComfyUI temp directory: {comfyui_temp}")
        except Exception as e:
            print(f"Could not remove {comfyui_temp}, trying to rename: {str(e)}")
            try:
                backup_name = f"{comfyui_temp}_backup_{unique_id}"
                os.rename(comfyui_temp, backup_name)
                print(f"Renamed {comfyui_temp} to {backup_name}")
                # Try to remove the renamed directory as well
                try:
                    shutil.rmtree(backup_name)
                    print(f"Removed renamed temp directory: {backup_name}")
                except:
                    pass
            except:
                print(f"Failed to rename {comfyui_temp}")
    
    print(f"Set up system temp directory: {temp_base_path}")
    return temp_base_path

# Function to clean up everything when the module exits
def module_cleanup():
    """Clean up all resources when the module is unloaded"""
    global MODULE_TEMP_DIR
    
    # Clean up our module temp directory
    if MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
        try:
            shutil.rmtree(MODULE_TEMP_DIR, ignore_errors=True)
            print(f"Cleaned up module temp directory: {MODULE_TEMP_DIR}")
        except:
            pass
    
    # Do a final sweep for any ComfyUI temp directories
    cleanup_comfyui_temp_directories()

# Call this before anything else
MODULE_TEMP_DIR = init_temp_directories()

# Register the cleanup handler to run when Python exits
import atexit
atexit.register(module_cleanup)

# Now import regular dependencies
import math
import torch
import random
import torchaudio
import folder_paths
import numpy as np
import platform
import subprocess
import importlib.util
import importlib.machinery
import argparse
from omegaconf import OmegaConf
from PIL import Image
from decimal import Decimal, ROUND_UP
import requests
from tqdm import tqdm

# Modify folder_paths module to use our temp directory
if hasattr(folder_paths, "get_temp_directory"):
    original_get_temp = folder_paths.get_temp_directory
    folder_paths.get_temp_directory = lambda: MODULE_TEMP_DIR
else:
    # Add the function if it doesn't exist
    setattr(folder_paths, 'get_temp_directory', lambda: MODULE_TEMP_DIR)

def import_inference_script(script_path):
    """Import a Python file as a module using its file path."""
    if not os.path.exists(script_path):
        raise ImportError(f"Script not found: {script_path}")

    module_name = "latentsync_inference"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {str(e)}")

    return module

def check_ffmpeg():
    try:
        if platform.system() == "Windows":
            # Check if ffmpeg exists in PATH
            ffmpeg_path = shutil.which("ffmpeg.exe")
            if ffmpeg_path is None:
                # Look for ffmpeg in common locations
                possible_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "ffmpeg", "bin"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        # Add to PATH
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                        return True
                print("FFmpeg not found. Please install FFmpeg and add it to PATH")
                return False
            return True
        else:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg")
        return False

def check_and_install_dependencies():
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found")

    required_packages = [
        'omegaconf',
        'pytorch_lightning',
        'transformers',
        'accelerate',
        'huggingface_hub',
        'einops',
        'diffusers',
        'ffmpeg-python' 
    ]

    def is_package_installed(package_name):
        return importlib.util.find_spec(package_name) is not None

    def install_package(package):
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', package],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            raise RuntimeError(f"Failed to install required package: {package}")

    for package in required_packages:
        if not is_package_installed(package):
            print(f"Installing required package: {package}")
            try:
                install_package(package)
            except Exception as e:
                print(f"Warning: Failed to install {package}: {str(e)}")
                raise

def normalize_path(path):
    """Normalize path to handle spaces and special characters"""
    return os.path.normpath(path).replace('\\', '/')

def get_ext_dir(subpath=None, mkdir=False):
    """Get extension directory path, optionally with a subpath"""
    # Get the directory containing this script
    dir = os.path.dirname(os.path.abspath(__file__))
    
    # Special case for temp directories
    if subpath and ("temp" in subpath.lower() or "tmp" in subpath.lower()):
        # Use our global temp directory instead
        global MODULE_TEMP_DIR
        sub_temp = os.path.join(MODULE_TEMP_DIR, subpath)
        if mkdir and not os.path.exists(sub_temp):
            os.makedirs(sub_temp, exist_ok=True)
        return sub_temp
    
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    return dir

def download_model(url, save_path):
    """Download a model from a URL and save it to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def pre_download_models():
    """Pre-download all required models."""
    models = {
        "s3fd-e19a316812.pth": "https://www.adrianbulat.com/downloads/python-fan/s3fd-e19a316812.pth",
        # Add other models here
    }

    cache_dir = os.path.join(MODULE_TEMP_DIR, "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    for model_name, url in models.items():
        save_path = os.path.join(cache_dir, model_name)
        if not os.path.exists(save_path):
            print(f"Downloading {model_name}...")
            download_model(url, save_path)
        else:
            print(f"{model_name} already exists in cache.")

def setup_models():
    """Setup and pre-download all required models."""
    # Use our global temp directory
    global MODULE_TEMP_DIR
    
    # Pre-download additional models
    pre_download_models()

    # Existing setup logic for LatentSync models
    cur_dir = get_ext_dir()
    ckpt_dir = os.path.join(cur_dir, "checkpoints")
    whisper_dir = os.path.join(ckpt_dir, "whisper")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)

    # Create a temp_downloads directory in our system temp
    temp_downloads = os.path.join(MODULE_TEMP_DIR, "downloads")
    os.makedirs(temp_downloads, exist_ok=True)
    
    unet_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
    whisper_path = os.path.join(whisper_dir, "tiny.pt")

    if not (os.path.exists(unet_path) and os.path.exists(whisper_path)):
        print("Downloading required model checkpoints... This may take a while.")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="ByteDance/LatentSync-1.5",
                             allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                             local_dir=ckpt_dir, 
                             local_dir_use_symlinks=False,
                             cache_dir=temp_downloads)
            print("Model checkpoints downloaded successfully!")
        except Exception as e:
            print(f"Error downloading models: {str(e)}")
            print("\nPlease download models manually:")
            print("1. Visit: https://huggingface.co/chunyu-li/LatentSync")
            print("2. Download: latentsync_unet.pt and whisper/tiny.pt")
            print(f"3. Place them in: {ckpt_dir}")
            print(f"   with whisper/tiny.pt in: {whisper_dir}")
            raise RuntimeError("Model download failed. See instructions above.")

class LatentSyncNode:
    def __init__(self):
        # Make sure our temp directory is the current one
        global MODULE_TEMP_DIR
        if not os.path.exists(MODULE_TEMP_DIR):
            os.makedirs(MODULE_TEMP_DIR, exist_ok=True)
        
        # Ensure ComfyUI temp doesn't exist
        comfyui_temp = "D:\\ComfyUI_windows\\temp"
        if os.path.exists(comfyui_temp):
            backup_name = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
            try:
                os.rename(comfyui_temp, backup_name)
            except:
                pass
        
        check_and_install_dependencies()
        setup_models()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "seed": ("INT", {"default": 1247}),
                    "lips_expression": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                    "inference_steps": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),
                 },}

    CATEGORY = "LatentSyncNode"

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio") 
    FUNCTION = "inference"

    def process_batch(self, batch, use_mixed_precision=False):
        """Process a batch of frames and save them to temporary files"""
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            processed_batch = batch.float() / 255.0
            if len(processed_batch.shape) == 3:
                processed_batch = processed_batch.unsqueeze(0)
            if processed_batch.shape[0] == 3:
                processed_batch = processed_batch.permute(1, 2, 0)
            if processed_batch.shape[-1] == 4:
                processed_batch = processed_batch[..., :3]
            return processed_batch

    def save_frames_to_files(self, frames, temp_dir):
        """Save frames to temporary files and return the list of file paths"""
        import cv2
        import numpy as np
        
        print(f"[FRAMES] Starting frame processing: shape={frames.shape}")
        
        frame_paths = []
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[FRAMES] Created frame directory: {temp_dir}")
        
        # Process frames in smaller batches to avoid memory issues
        batch_size = 10  # Process 10 frames at a time
        total_frames = frames.shape[0]
        print(f"[FRAMES] Total frames to process: {total_frames}, batch_size={batch_size}")
        
        # Calculate total number of batches
        total_batches = (total_frames + batch_size - 1) // batch_size
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, total_frames, batch_size), 
                     total=total_batches,
                     desc="Processing frames",
                     unit="batch"):
            end_idx = min(i + batch_size, total_frames)
            batch = frames[i:end_idx]
            print(f"[FRAMES] Processing batch {i}-{end_idx} of {total_frames}")
            
            for j, frame in enumerate(batch):
                frame_idx = i + j
                # Convert tensor to numpy array
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                
                # Save frame to file
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
                
                if frame_idx % 50 == 0:
                    print(f"[FRAMES] Saved frame {frame_idx}/{total_frames}")
                
                # Clear memory
                del frame_np
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Clear batch memory
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[FRAMES] Memory cleared after batch {i}-{end_idx}")
        
        print(f"[FRAMES] Completed saving {len(frame_paths)} frames")
        return frame_paths

    def process_audio_stream(self, waveform, sample_rate, target_sample_rate=16000, chunk_size=500000):
        """Process audio in chunks and save to temporary file"""
        import tempfile
        import soundfile as sf
        
        print(f"[AUDIO] Starting audio processing: shape={waveform.shape}, sample_rate={sample_rate}")
        
        # Create temporary file for processed audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        print(f"[AUDIO] Created temporary audio file: {temp_audio_path}")
        
        # Initialize resampler
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        print(f"[AUDIO] Initialized resampler: {sample_rate}Hz -> {target_sample_rate}Hz")
        
        # Process audio in chunks
        total_samples = waveform.shape[1]
        processed_chunks = []
        print(f"[AUDIO] Total samples to process: {total_samples}")
        
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk = waveform[:, i:end_idx]
            print(f"[AUDIO] Processing chunk {i}-{end_idx} of {total_samples}")
            
            # Process chunk
            chunk_resampled = resampler(chunk)
            processed_chunks.append(chunk_resampled)
            
            # Clear memory
            del chunk
            del chunk_resampled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[AUDIO] Memory cleared after chunk {i}-{end_idx}")
        
        # Concatenate and save
        print("[AUDIO] Concatenating processed chunks")
        processed_audio = torch.cat(processed_chunks, dim=1)
        print(f"[AUDIO] Saving processed audio: shape={processed_audio.shape}")
        torchaudio.save(temp_audio_path, processed_audio, target_sample_rate)
        print(f"[AUDIO] Saved processed audio to {temp_audio_path}")
        
        # Clear memory
        del processed_chunks
        del processed_audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[AUDIO] Final memory cleanup completed")
        
        return temp_audio_path, target_sample_rate

    def inference(self, images, audio, seed, lips_expression=1.5, inference_steps=20):
        # Use our module temp directory
        global MODULE_TEMP_DIR
        
        print("[INFERENCE] Starting inference process")
        print(f"[INFERENCE] Input images shape: {images.shape if hasattr(images, 'shape') else 'list of tensors'}")
        print(f"[INFERENCE] Audio shape: {audio['waveform'].shape}, Sample rate: {audio['sample_rate']}")
        
        # Initialize logging function
        def log_memory(message):
            if torch.cuda.is_available():
                try:
                    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    free = total_mem - reserved
                    print(f"[MEMORY] {message}: Total: {total_mem:.2f}GB, Reserved: {reserved:.2f}GB, Allocated: {allocated:.2f}GB, Free: {free:.2f}GB")
                except Exception as e:
                    print(f"[MEMORY] Error logging memory: {str(e)}")
            else:
                print(f"[LOG] {message}")
        
        # Start logging
        log_memory("Starting inference")
        
        # Create a run-specific subdirectory in our temp directory
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(MODULE_TEMP_DIR, f"run_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[INFERENCE] Created temporary directory: {temp_dir}")
        
        # Create frame directory
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"[INFERENCE] Created frames directory: {frames_dir}")
        
        try:
            # Process input frames and save to files
            log_memory("Before processing input frames")
            if isinstance(images, list):
                frames = torch.stack(images)
                print(f"[INFERENCE] Stacked list of images into tensor of shape {frames.shape}")
            else:
                frames = images  # No need to clone since we're saving to files
                print(f"[INFERENCE] Using image tensor of shape {frames.shape}")
            
            # Save frames to files
            frame_paths = self.save_frames_to_files(frames, frames_dir)
            print(f"[INFERENCE] Saved {len(frame_paths)} frames to {frames_dir}")
            
            # Clear frames from memory
            del frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory("After saving frames to files")
            
            # Process audio in streaming mode
            log_memory("Before processing audio")
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
                print(f"[INFERENCE] Squeezed waveform to shape {waveform.shape}")
            
            # Process audio in streaming mode and save to file
            audio_path, new_sample_rate = self.process_audio_stream(
                waveform, 
                sample_rate,
                target_sample_rate=16000 if sample_rate != 16000 else sample_rate
            )
            print(f"[INFERENCE] Processed audio saved to {audio_path}")
            
            # Clear audio from memory
            del waveform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory("After processing audio")
            
            # Create video from frame files
            temp_video_path = os.path.join(temp_dir, f"temp_{run_id}.mp4")
            output_video_path = os.path.join(temp_dir, f"latentsync_{run_id}_out.mp4")
            
            # Create video from frame files using ffmpeg
            import subprocess
            frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", "25",
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                temp_video_path
            ]
            
            print(f"[INFERENCE] Creating video from frames using ffmpeg")
            print(f"[INFERENCE] Command: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"[INFERENCE] Video created at {temp_video_path}")
            
            # Clear frame files to save space
            print("[INFERENCE] Cleaning up frame files")
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            try:
                os.rmdir(frames_dir)
            except:
                pass
            print("[INFERENCE] Frame files cleanup completed")
            
            # Get the extension directory
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"[INFERENCE] Current directory: {cur_dir}")
            
            # Define paths to required files and configs
            inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
            config_path = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
            scheduler_config_path = os.path.join(cur_dir, "configs")
            ckpt_path = os.path.join(cur_dir, "checkpoints", "latentsync_unet.pt")
            whisper_ckpt_path = os.path.join(cur_dir, "checkpoints", "whisper", "tiny.pt")
            
            print(f"[INFERENCE] Loading config from {config_path}")
            config = OmegaConf.load(config_path)
            
            # Set the correct mask image path
            mask_image_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
            if not os.path.exists(mask_image_path):
                alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
                if os.path.exists(alt_mask_path):
                    mask_image_path = alt_mask_path
                    print(f"[INFERENCE] Using alternative mask path: {mask_image_path}")
                else:
                    print(f"[WARNING] Could not find mask image at expected locations")
            
            # Set mask path in config
            if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
                config.data.mask_image_path = mask_image_path
                print(f"[INFERENCE] Set mask path in config: {mask_image_path}")
            
            print("[INFERENCE] Creating inference arguments")
            args = argparse.Namespace(
                unet_config_path=config_path,
                inference_ckpt_path=ckpt_path,
                video_path=temp_video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                inference_steps=inference_steps,
                guidance_scale=lips_expression,
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path,
                device=torch.device('cpu'),  # Force CPU to save memory
                batch_size=1,  # Reduce batch size to save memory
                use_mixed_precision=True,  # Enable mixed precision
                temp_dir=temp_dir,
                mask_image_path=mask_image_path
            )
            
            # Set PYTHONPATH
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)
            print(f"[INFERENCE] Updated PYTHONPATH with: {package_root}, {cur_dir}")
            
            # Import the inference module
            print(f"[INFERENCE] Importing inference module from {inference_script_path}")
            inference_module = import_inference_script(inference_script_path)
            
            # Run inference
            print("[INFERENCE] Starting main inference process")
            inference_module.main(config, args)
            print("[INFERENCE] Main inference process completed")
            
            # Read output video in chunks
            print(f"[INFERENCE] Reading output video from {output_video_path}")
            try:
                import cv2
                cap = cv2.VideoCapture(output_video_path)
                frames_list = []
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert from BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(torch.from_numpy(frame_rgb).float() / 255.0)
                    frame_count += 1
                    
                    if frame_count % 50 == 0:
                        print(f"[INFERENCE] Processed {frame_count} frames")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                cap.release()
                print(f"[INFERENCE] Completed reading {frame_count} frames")
                processed_frames = torch.stack(frames_list)
                print(f"[INFERENCE] Stacked frames into tensor of shape {processed_frames.shape}")
            except Exception as e:
                print(f"[ERROR] Error reading output video: {str(e)}")
                raise
            
            # Package audio for return
            try:
                # Read the actual audio from the file
                waveform, sample_rate = torchaudio.load(audio_path)
                # Add batch dimension to waveform [batch_size, channels, samples]
                waveform = waveform.unsqueeze(0)
                resampled_audio = {
                    "waveform": waveform,
                    "sample_rate": sample_rate
                }
            except Exception as e:
                print(f"[WARNING] Failed to read audio from {audio_path}: {str(e)}")
                # Fallback to placeholder if reading fails
                resampled_audio = {
                    "waveform": torch.zeros((1, 1, 1)),  # Placeholder with correct dimensions
                    "sample_rate": new_sample_rate
                }
            print("[INFERENCE] Packaging results for return")
            
            # Ensure audio and video are on CPU before returning
            if torch.cuda.is_available():
                if hasattr(resampled_audio["waveform"], 'device') and resampled_audio["waveform"].device.type == 'cuda':
                    resampled_audio["waveform"] = resampled_audio["waveform"].cpu()
                if hasattr(processed_frames, 'device') and processed_frames.device.type == 'cuda':
                    processed_frames = processed_frames.cpu()
            
            return (processed_frames, resampled_audio)
            
        except Exception as e:
            print(f"[ERROR] Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            print("[INFERENCE] Starting cleanup")
            # Clean up temporary files
            for path in [temp_video_path, output_video_path, audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"[INFERENCE] Removed temporary file: {path}")
                    except:
                        print(f"[WARNING] Failed to remove temporary file: {path}")
            
            # Remove temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"[INFERENCE] Removed temporary directory: {temp_dir}")
                except:
                    print(f"[WARNING] Failed to remove temporary directory: {temp_dir}")
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[INFERENCE] Final memory cleanup completed")
            print("[INFERENCE] Cleanup completed")

class VideoLengthAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1}),
            }
        }

    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "adjust"

    def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = int(audio["sample_rate"])
        original_frames = [images[i] for i in range(images.shape[0])] if isinstance(images, torch.Tensor) else images.copy()

        if mode == "normal":
            # Add silent padding to the audio and then trim video to match
            audio_duration = waveform.shape[1] / sample_rate
            
            # Add silent padding to the audio
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            
            # Calculate required frames based on the padded audio
            padded_audio_duration = (waveform.shape[1] + silence_samples) / sample_rate
            required_frames = int(padded_audio_duration * fps)
            
            if len(original_frames) > required_frames:
                # Trim video frames to match padded audio duration
                adjusted_frames = original_frames[:required_frames]
            else:
                # If video is shorter than padded audio, keep all video frames
                # and trim the audio accordingly
                adjusted_frames = original_frames
                required_samples = int(len(original_frames) / fps * sample_rate)
                padded_audio = padded_audio[:, :required_samples]
            
            return (
                torch.stack(adjusted_frames),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )
            
            # This return statement is no longer needed as it's handled in the updated code

        elif mode == "pingpong":
            video_duration = len(original_frames) / fps
            audio_duration = waveform.shape[1] / sample_rate
            if audio_duration <= video_duration:
                required_samples = int(video_duration * sample_rate)
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)

                return (
                    torch.stack(original_frames),
                    {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

            else:
                silence_samples = math.ceil(silent_padding_sec * sample_rate)
                silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
                padded_audio = torch.cat([waveform, silence], dim=1)
                total_duration = (waveform.shape[1] + silence_samples) / sample_rate
                target_frames = math.ceil(total_duration * fps)
                reversed_frames = original_frames[::-1][1:-1]  # Remove endpoints
                frames = original_frames + reversed_frames
                while len(frames) < target_frames:
                    frames += frames[:target_frames - len(frames)]
                return (
                    torch.stack(frames[:target_frames]),
                    {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

        elif mode == "loop_to_audio":
            # Add silent padding then simple loop
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            total_duration = (waveform.shape[1] + silence_samples) / sample_rate
            target_frames = math.ceil(total_duration * fps)

            frames = original_frames.copy()
            while len(frames) < target_frames:
                frames += original_frames[:target_frames - len(frames)]
            
            return (
                torch.stack(frames[:target_frames]),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )



# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LatentSyncNode": LatentSyncNode,
    "VideoLengthAdjuster": VideoLengthAdjuster,
}

# Display Names for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSyncNode": "LatentSync1.5 Node",
    "VideoLengthAdjuster": "Video Length Adjuster",
 }