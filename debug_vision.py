import cv2
import numpy as np
import rawpy
import os

# Set up clean project paths inside your Latenide workspace
BASE_DIR = r"C:\Users\elahi\Documents\Latenide"
INGEST_DIR = os.path.join(BASE_DIR, "Camera_Ingest_Test")

# Simplified paths looking for your renamed test files
SHARP_IMAGE_PATH = os.path.join(INGEST_DIR, "sharp.ARW")
BLURRY_IMAGE_PATH = os.path.join(INGEST_DIR, "blurry.ARW")

# Debug output folder
OUTPUT_DIR = os.path.join(BASE_DIR, "debug_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect_image(file_path, label):
    if not os.path.exists(file_path):
        print(f"❌ Cannot find file: {os.path.basename(file_path)} inside Camera_Ingest_Test")
        return
        
    print(f"📖 Safely reading: {os.path.basename(file_path)}...")
    with rawpy.imread(file_path) as raw:
        thumb = raw.extract_thumb()
        img_array = np.frombuffer(thumb.data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Save a direct view of the source thumbnail inside Latenide/debug_output
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"debug_{label}_source.jpg"), img)
        
        # Run processing layers
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Extract edge matrix
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges_visual = np.uint8(np.absolute(edges))
        
        # Save the edge map visualization inside Latenide/debug_output
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"debug_{label}_edges.jpg"), edges_visual)
        
        score = edges.var()
        print(f"📊 {label} Resolution: {img.shape[1]}x{img.shape[0]} | Variance Score: {score:.2f}\n")

if __name__ == "__main__":
    print("=============================================")
    print("      RUNNING ISOLATED VISION DIAGNOSTICS   ")
    print("=============================================\n")
    
    inspect_image(SHARP_IMAGE_PATH, "Target_Sharp")
    inspect_image(BLURRY_IMAGE_PATH, "Target_Blurry")
    
    print(f"📁 Analysis complete. Check files inside: {OUTPUT_DIR}")