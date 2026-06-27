import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Load hidden parameters from your .env workspace environment
from dotenv import load_dotenv
load_dotenv()

# Connect your local functional modules
import pipeline_core

RAW_EXTENSIONS = {'.arw', '.cr3', '.nef', '.dng'}


def build_config_from_env():
    """Read optional pipeline settings from environment variables."""
    config = pipeline_core.default_config()
    config["threshold"] = float(os.environ.get("LATENIDE_THRESHOLD", config["threshold"]))
    config["reject_closed_eye"] = (
        os.environ.get("LATENIDE_REJECT_CLOSED_EYE", "0") == "1"
    )
    return config


class RawImageHandler(FileSystemEventHandler):

    def __init__(self, config=None, log=print):
        super().__init__()
        # Shared config (carries seen_hashes across assets for burst grouping).
        self.config = config or build_config_from_env()
        self.log = log

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        _, extension = os.path.splitext(file_path)

        if extension.lower() in RAW_EXTENSIONS:
            self.log(f"\n[NEW ASSET DETECTED]: {os.path.basename(file_path)}")
            self.process_pipeline_trigger(file_path)

    def process_pipeline_trigger(self, file_path):
        base_dir = os.path.dirname(file_path)
        result = pipeline_core.process_asset(
            file_path, base_dir, config=self.config, log=self.log
        )
        # Daemon keeps the cache lean: drop the temporary preview JPEG.
        preview = result.get("preview_path")
        if preview and os.path.exists(preview):
            os.remove(preview)
        return result


if __name__ == "__main__":
    WATCH_DIRECTORY = os.path.expanduser("~/Documents/Latenide/Camera_Ingest_Test")
    os.makedirs(WATCH_DIRECTORY, exist_ok=True)
    
    event_handler = RawImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIRECTORY, recursive=False)
    
    print(f"🚀 Core Watcher Engine Active. Monitoring: {WATCH_DIRECTORY}")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Watcher Engine gracefully...")
        observer.stop()
        
    observer.join()