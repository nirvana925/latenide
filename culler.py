import cv2
import numpy as np
import rawpy
import os
import shutil

def evaluate_asset(file_path, threshold=20.0):
    """
    Extracts embedded preview, cleans pixel noise, and uses 
    variance calculation to score image sharpness.
    """
    try:
        with rawpy.imread(file_path) as raw:
            try:
                thumb = raw.extract_thumb()
            except:
                print(f" ❌ No embedded preview found.")
                return False, 0.0

            if thumb.format == rawpy.ThumbFormat.JPEG:
                img_array = np.frombuffer(thumb.data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                return False, 0.0

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply gentle blur to isolate true structural focus from noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Revert to Variance calculation (highly reliable separation)
        score = cv2.Laplacian(blurred, cv2.CV_64F).var()
        
        is_sharp = score >= threshold
        return is_sharp, score

    except Exception as e:
        print(f" ❌ Error processing matrix: {e}")
        return False, 0.0

def route_asset(file_path, is_sharp, score):
    base_dir = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    if is_sharp:
        destination_folder = os.path.join(base_dir, "01_Sharp_Queue")
        status = "PASSED"
    else:
        destination_folder = os.path.join(base_dir, "02_Rejects_Queue")
        status = "REJECTED (BLURRY)"
        
    os.makedirs(destination_folder, exist_ok=True)
    shutil.move(file_path, os.path.join(destination_folder, filename))
    print(f" 🧭 [ROUTING SYSTEM]: {status} (Score: {score:.2f}) -> {os.path.basename(destination_folder)}")