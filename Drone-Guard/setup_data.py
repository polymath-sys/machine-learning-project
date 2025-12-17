import os
import shutil
from PIL import Image
from tqdm import tqdm # pip install tqdm if you don't have it

# Configuration
SOURCE_ROOT = "UCSD_Anomaly_Dataset.v1p2/UCSDped2"
DEST_ROOT = "data/ped2"

def process_dataset(source_type, dest_type):
    """
    Moves and converts images from source_type (e.g., 'Train') 
    to dest_type (e.g., 'training/frames')
    """
    source_path = os.path.join(SOURCE_ROOT, source_type)
    dest_path = os.path.join(DEST_ROOT, dest_type, "frames")
    
    if not os.path.exists(source_path):
        print(f"Error: Could not find {source_path}")
        return

    # Get all video folders (e.g., Train001, Train002...)
    video_folders = []
    for f in os.listdir(source_path):
        # 1. Skip hidden files (like .DS_Store on Mac)
        if f.startswith('.'):
            continue
        
        full_path = os.path.join(source_path, f)
        
        # 2. **CRITICAL FIX:** Only include items that are actual directories
        if os.path.isdir(full_path):
            video_folders.append(f)
            
    video_folders.sort()
    
    print(f"Processing {source_type} data...")
    
    for folder in tqdm(video_folders):
        src_video_path = os.path.join(source_path, folder)
        
        # Rename folder to match repo convention (Train001 -> video_01) (Optional but clean)
        # But keeping original names is often safer for some dataloaders. 
        # Let's just keep the folder name but move it.
        dst_video_path = os.path.join(dest_path, folder)
        
        os.makedirs(dst_video_path, exist_ok=True)
        
        # Convert and move images
        images = sorted([img for img in os.listdir(src_video_path) if img.endswith('.tif')])
        
        for img_name in images:
            # Open .tif
            with Image.open(os.path.join(src_video_path, img_name)) as img:
                # Convert to RGB (jpeg doesn't support some tif modes)
                rgb_im = img.convert('RGB')
                
                # Save as .jpg
                new_name = img_name.replace('.tif', '.jpg')
                rgb_im.save(os.path.join(dst_video_path, new_name), quality=95)

if __name__ == "__main__":
    # Create main directories
    os.makedirs(os.path.join(DEST_ROOT, "training", "frames"), exist_ok=True)
    os.makedirs(os.path.join(DEST_ROOT, "testing", "frames"), exist_ok=True)

    # Process Train -> training
    process_dataset("Train", "training")
    
    # Process Test -> testing
    process_dataset("Test", "testing")
    
    print("\n✅ Success! Data is ready at: Drone-Guard/data/ped2")