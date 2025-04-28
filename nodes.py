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
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            processed_batch = batch.float() / 255.0
            if len(processed_batch.shape) == 3:
                processed_batch = processed_batch.unsqueeze(0)
            if processed_batch.shape[0] == 3:
                processed_batch = processed_batch.permute(1, 2, 0)
            if processed_batch.shape[-1] == 4:
                processed_batch = processed_batch[..., :3]
            return processed_batch

    def inference(self, images, audio, seed, lips_expression=1.5, inference_steps=20):
        # Use our module temp directory
        global MODULE_TEMP_DIR
        
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
        
        # Get GPU capabilities and memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {device}")
        BATCH_SIZE = 4
        use_mixed_precision = False
        
        # Determine available GPU memory and set parameters accordingly
        if torch.cuda.is_available():
            # Clear GPU cache before we start
            torch.cuda.empty_cache()
            log_memory("After initial cache clear")
            
            # Get total and free memory
            total_mem = torch.cuda.get_device_properties(0).total_memory
            # Convert to GB for better readability in logs
            total_mem_gb = total_mem / (1024 ** 3)
            
            # Try to get current free memory
            try:
                reserved_mem = torch.cuda.memory_reserved(0)
                allocated_mem = torch.cuda.memory_allocated(0)
                free_mem = total_mem - reserved_mem
                free_mem_gb = free_mem / (1024 ** 3)
                print(f"[INFO] Total GPU memory: {total_mem_gb:.2f}GB, Free: {free_mem_gb:.2f}GB")
            except:
                # If we can't get free memory, just use total memory
                free_mem_gb = total_mem_gb
                print(f"[INFO] Total GPU memory: {total_mem_gb:.2f}GB")
                
            # Determine batch size and precision based on available memory
            if free_mem_gb > 12:  # High-end GPUs with plenty of memory
                BATCH_SIZE = 8
                use_mixed_precision = True
                print("[CONFIG] Using high-memory settings: batch_size=8, mixed_precision=True")
            elif free_mem_gb > 6:  # Mid-range GPUs
                BATCH_SIZE = 4
                use_mixed_precision = True
                print("[CONFIG] Using mid-memory settings: batch_size=4, mixed_precision=True")
            else:  # Lower memory GPUs
                BATCH_SIZE = 2
                use_mixed_precision = True
                print("[CONFIG] Using low-memory settings: batch_size=2, mixed_precision=True")
            
            # Set performance options
            torch.backends.cudnn.benchmark = True
            if free_mem_gb > 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[CONFIG] Enabled TF32 precision")
            
            # Set memory fraction to avoid using all GPU memory
            try:
                torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
                print("[CONFIG] Set GPU memory fraction to 0.7")
            except:
                print("[WARNING] Could not set memory fraction, using default memory management")

        # Create a run-specific subdirectory in our temp directory
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(MODULE_TEMP_DIR, f"run_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[INFO] Created temporary directory: {temp_dir}")
        
        # Ensure ComfyUI temp doesn't exist again (in case something recreated it)
        comfyui_temp = get_comfyui_temp_dir()
        if comfyui_temp and os.path.exists(comfyui_temp):
            backup_name = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
            try:
                os.rename(comfyui_temp, backup_name)
                print(f"[INFO] Renamed ComfyUI temp directory to {backup_name}")
            except:
                print(f"[WARNING] Failed to rename ComfyUI temp directory")
        
        temp_video_path = None
        output_video_path = None
        audio_path = None

        try:
            # Create temporary file paths in our system temp directory
            temp_video_path = os.path.join(temp_dir, f"temp_{run_id}.mp4")
            output_video_path = os.path.join(temp_dir, f"latentsync_{run_id}_out.mp4")
            audio_path = os.path.join(temp_dir, f"latentsync_{run_id}_audio.wav")
            
            # Get the extension directory
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Log image information
            print(f"[INFO] Input images shape: {images.shape if hasattr(images, 'shape') else 'list of tensors'}")
            print(f"[INFO] Audio shape: {audio['waveform'].shape}, Sample rate: {audio['sample_rate']}")
            
            # Process input frames - keep on CPU as long as possible
            log_memory("Before processing input frames")
            if isinstance(images, list):
                frames = torch.stack(images)
                print(f"[INFO] Stacked list of images into tensor of shape {frames.shape}")
            else:
                frames = images.clone()  # Use clone to avoid modifying original data
                print(f"[INFO] Cloned image tensor of shape {frames.shape}")
                
            # Convert to bytes while still on CPU
            frames = (frames * 255).byte()
            print(f"[INFO] Converted frames to byte tensor")
            
            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)
                print(f"[INFO] Unsqueezed frames to shape {frames.shape}")
            
            log_memory("After processing input frames")
                
            # Process audio - keep on CPU initially
            log_memory("Before processing audio")
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
                print(f"[INFO] Squeezed waveform to shape {waveform.shape}")

            if sample_rate != 16000:
                print(f"[INFO] Resampling audio from {sample_rate}Hz to 16000Hz")
                # Create resampler on CPU first, then move to device only for processing
                new_sample_rate = 16000
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=new_sample_rate
                )
                
                if device.type == 'cuda':
                    # Process in smaller chunks if needed for large audio
                    if waveform.shape[1] > 1000000:  # If very long audio
                        print("[INFO] Processing large audio in chunks")
                        chunk_size = 500000
                        chunks = []
                        
                        for i in range(0, waveform.shape[1], chunk_size):
                            end = min(i + chunk_size, waveform.shape[1])
                            chunk = waveform[:, i:end]
                            print(f"[INFO] Processing audio chunk {i}-{end} of {waveform.shape[1]}")
                            # Move each chunk to GPU, process, then back to CPU
                            chunk_gpu = chunk.to(device)
                            log_memory(f"After moving audio chunk {i}-{end} to GPU")
                            chunk_resampled = resampler.to(device)(chunk_gpu)
                            chunks.append(chunk_resampled.cpu())
                            # Clear GPU memory after each chunk
                            del chunk_gpu
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                log_memory(f"After processing audio chunk {i}-{end}")
                                
                        waveform_16k = torch.cat(chunks, dim=1)
                        print(f"[INFO] Concatenated {len(chunks)} audio chunks")
                    else:
                        # For smaller audio, process normally
                        print("[INFO] Processing audio on GPU")
                        waveform_gpu = waveform.to(device)
                        log_memory("After moving audio to GPU")
                        waveform_16k = resampler.to(device)(waveform_gpu)
                        waveform_16k = waveform_16k.cpu()  # Move back to CPU
                        print("[INFO] Moved resampled audio back to CPU")
                        del waveform_gpu
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            log_memory("After audio processing")
                else:
                    # For CPU processing
                    print("[INFO] Processing audio on CPU")
                    waveform_16k = resampler(waveform)
                    
                waveform, sample_rate = waveform_16k, new_sample_rate
                print(f"[INFO] Resampled audio shape: {waveform.shape}, New sample rate: {sample_rate}")

            # Package resampled audio
            resampled_audio = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate
            }
            
            # Save audio to file
            print(f"[INFO] Saving audio to {audio_path}")
            torchaudio.save(audio_path, waveform, sample_rate)

            # Save video to file - process in batches if needed
            log_memory("Before saving video to file")
            try:
                import torchvision.io as io
                # For large videos, write frames in batches to avoid OOM
                if frames.shape[0] > 100:  # If video has many frames
                    import cv2
                    print(f"[INFO] Processing large video with {frames.shape[0]} frames in batches")
                    
                    # Get video dimensions
                    height, width = frames.shape[1], frames.shape[2]
                    print(f"[INFO] Video dimensions: {width}x{height}")
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(temp_video_path, fourcc, 25, (width, height))
                    
                    # Process in batches
                    batch_size = min(50, frames.shape[0])  # Process 50 frames at a time
                    print(f"[INFO] Processing video in batches of {batch_size} frames")
                    for i in range(0, frames.shape[0], batch_size):
                        end_idx = min(i + batch_size, frames.shape[0])
                        batch_frames = frames[i:end_idx]
                        print(f"[INFO] Processing frames {i}-{end_idx} of {frames.shape[0]}")
                        
                        # Convert to OpenCV format
                        for frame in batch_frames:
                            # Convert from RGB to BGR for OpenCV
                            frame_np = frame.numpy()
                            frame_cv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame_cv)
                            
                    video_writer.release()
                    print(f"[INFO] Finished writing video to {temp_video_path}")
                else:
                    # For smaller videos, use torchvision directly
                    print(f"[INFO] Writing video with {frames.shape[0]} frames using torchvision")
                    io.write_video(temp_video_path, frames, fps=25, video_codec='h264')
                    print(f"[INFO] Finished writing video to {temp_video_path}")
            except Exception as e:
                print(f"[ERROR] Error writing video with torchvision: {str(e)}")
                # Fallback to pyav if torchvision fails
                try:
                    import av
                    print("[INFO] Falling back to PyAV for video writing")
                    container = av.open(temp_video_path, mode='w')
                    stream = container.add_stream('h264', rate=25)
                    stream.width = frames.shape[2]
                    stream.height = frames.shape[1]

                    for i, frame in enumerate(frames):
                        if i % 50 == 0:
                            print(f"[INFO] Writing frame {i}/{frames.shape[0]}")
                        frame = av.VideoFrame.from_ndarray(frame.numpy(), format='rgb24')
                        packet = stream.encode(frame)
                        container.mux(packet)

                    packet = stream.encode(None)
                    container.mux(packet)
                    container.close()
                    print(f"[INFO] Finished writing video to {temp_video_path} using PyAV")
                except Exception as e:
                    print(f"[ERROR] Error writing video with PyAV: {str(e)}")
                    raise

            # Clear memory after video/audio processing
            del frames
            del waveform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log_memory("After clearing video/audio memory")

            # Define paths to required files and configs
            inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
            config_path = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
            scheduler_config_path = os.path.join(cur_dir, "configs")
            ckpt_path = os.path.join(cur_dir, "checkpoints", "latentsync_unet.pt")
            whisper_ckpt_path = os.path.join(cur_dir, "checkpoints", "whisper", "tiny.pt")

            # Create config and args
            config = OmegaConf.load(config_path)

            # Set the correct mask image path
            mask_image_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
            # Make sure the mask image exists
            if not os.path.exists(mask_image_path):
                # Try to find it in the utils directory directly
                alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
                if os.path.exists(alt_mask_path):
                    mask_image_path = alt_mask_path
                else:
                    print(f"Warning: Could not find mask image at expected locations")

            # Set mask path in config
            if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
                config.data.mask_image_path = mask_image_path

            args = argparse.Namespace(
                unet_config_path=config_path,
                inference_ckpt_path=ckpt_path,
                video_path=temp_video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                inference_steps=inference_steps,
                guidance_scale=lips_expression,  # Using lips_expression for the guidance_scale
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path,
                device=device,
                batch_size=BATCH_SIZE,
                use_mixed_precision=use_mixed_precision,
                temp_dir=temp_dir,
                mask_image_path=mask_image_path
            )

            # Set PYTHONPATH to include our directories 
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)

            # Clean GPU cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Check and prevent ComfyUI temp creation again
            if os.path.exists(comfyui_temp):
                try:
                    os.rename(comfyui_temp, f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}")
                except:
                    pass

            # Import the inference module
            inference_module = import_inference_script(inference_script_path)
            
            # Monkey patch any temp directory functions in the inference module
            if hasattr(inference_module, 'get_temp_dir'):
                inference_module.get_temp_dir = lambda *args, **kwargs: temp_dir
                
            # Create subdirectories that the inference module might expect
            inference_temp = os.path.join(temp_dir, "temp")
            os.makedirs(inference_temp, exist_ok=True)
            
            # Run inference
            inference_module.main(config, args)

            # Clean GPU cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Verify output file exists
            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"Output video not found at: {output_video_path}")
            
            # Read the processed video - ensure it's loaded as CPU tensor
            # Read video in chunks to avoid OOM
            try:
                processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]
                processed_frames = processed_frames.float() / 255.0
            except Exception as e:
                print(f"Error reading output video with torchvision: {str(e)}")
                # Fallback to reading with OpenCV if torchvision fails
                try:
                    import cv2
                    cap = cv2.VideoCapture(output_video_path)
                    frames_list = []
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Convert from BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames_list.append(torch.from_numpy(frame_rgb).float() / 255.0)
                    
                    cap.release()
                    processed_frames = torch.stack(frames_list)
                except Exception as e:
                    print(f"Error reading output video with OpenCV: {str(e)}")
                    raise

            # Ensure audio is on CPU before returning
            if torch.cuda.is_available():
                if hasattr(resampled_audio["waveform"], 'device') and resampled_audio["waveform"].device.type == 'cuda':
                    resampled_audio["waveform"] = resampled_audio["waveform"].cpu()
                if hasattr(processed_frames, 'device') and processed_frames.device.type == 'cuda':
                    processed_frames = processed_frames.cpu()

            return (processed_frames, resampled_audio)

        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Clean up temporary files individually
            for path in [temp_video_path, output_video_path, audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Removed temporary file: {path}")
                    except Exception as e:
                        print(f"Failed to remove {path}: {str(e)}")

            # Remove temporary run directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Removed run temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Failed to remove temp run directory: {str(e)}")

            # Clean up any ComfyUI temp directories again (in case they were created during execution)
            cleanup_comfyui_temp_directories()

            # Final GPU cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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