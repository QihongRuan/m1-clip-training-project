#!/usr/bin/env python3
"""
M1-Optimized CLIP Training Launcher
Main execution script for multimodal training
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

class CLIPTrainingLauncher:
    """Manage CLIP training and monitoring processes"""
    
    def __init__(self):
        self.training_process = None
        self.monitor_process = None
        self.project_dir = Path(__file__).parent
        
    def check_dependencies(self):
        """Check if all required files are present"""
        required_files = [
            'm1_clip_model.py',
            'm1_clip_dataset.py', 
            'm1_clip_training.py',
            'clip_monitor.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return False
        
        print("✅ All required files found")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        dirs = ['clip_checkpoints', 'clip_data', 'logs']
        
        for dir_name in dirs:
            dir_path = self.project_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"📁 Created directory: {dir_name}")
    
    def start_training(self):
        """Start CLIP training process"""
        print("\n🚀 Starting CLIP training process...")
        
        training_script = self.project_dir / 'm1_clip_training.py'
        log_file = self.project_dir / 'logs' / 'training.log'
        
        try:
            with open(log_file, 'w') as f:
                self.training_process = subprocess.Popen(
                    [sys.executable, str(training_script)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=self.project_dir
                )
            
            print(f"✅ Training started (PID: {self.training_process.pid})")
            print(f"📝 Logs: {log_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start training: {e}")
            return False
    
    def start_monitoring(self):
        """Start monitoring process"""
        print("\n📊 Starting real-time monitoring...")
        
        monitor_script = self.project_dir / 'clip_monitor.py'
        
        try:
            # Run monitor in a new terminal window (macOS)
            if sys.platform == 'darwin':  # macOS
                apple_script = f'''
                tell application "Terminal"
                    do script "cd '{self.project_dir}' && python3 '{monitor_script}'"
                    activate
                end tell
                '''
                
                subprocess.run(['osascript', '-e', apple_script])
                print("✅ Monitor opened in new terminal window")
            else:
                # Fallback: run in background
                self.monitor_process = subprocess.Popen(
                    [sys.executable, str(monitor_script)],
                    cwd=self.project_dir
                )
                print(f"✅ Monitor started in background (PID: {self.monitor_process.pid})")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start monitor: {e}")
            return False
    
    def wait_for_completion(self):
        """Wait for training to complete"""
        print("\n⏳ Waiting for training to complete...")
        print("💡 You can monitor progress in the separate terminal window")
        print("⌨️  Press Ctrl+C to stop training")
        
        try:
            self.training_process.wait()
            print("\n🎉 Training completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self.cleanup()
    
    def cleanup(self):
        """Clean up processes"""
        print("\n🧹 Cleaning up processes...")
        
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            time.sleep(2)
            if self.training_process.poll() is None:
                self.training_process.kill()
            print("✅ Training process terminated")
        
        if self.monitor_process and self.monitor_process.poll() is None:
            self.monitor_process.terminate()
            time.sleep(1)
            if self.monitor_process.poll() is None:
                self.monitor_process.kill()
            print("✅ Monitor process terminated")
    
    def run(self):
        """Main execution flow"""
        print("🎨 M1-Optimized CLIP Training Launcher")
        print("=" * 50)
        print("🧠 Contrastive Language-Image Pre-training")
        print("🖥️  Optimized for M1 MacBook Pro")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Create directories
        self.create_directories()
        
        # Start training
        if not self.start_training():
            return False
        
        # Wait a bit for training to initialize
        time.sleep(3)
        
        # Start monitoring
        if not self.start_monitoring():
            print("⚠️  Monitor failed to start, but training continues")
        
        # Wait for completion
        try:
            self.wait_for_completion()
        finally:
            self.cleanup()
        
        print("\n🎯 CLIP training session completed!")
        print("📊 Check ./clip_checkpoints/ for results")
        return True

def main():
    """Main function"""
    launcher = CLIPTrainingLauncher()
    success = launcher.run()
    
    if success:
        print("\n✨ Session completed successfully!")
    else:
        print("\n❌ Session ended with errors")
        sys.exit(1)

if __name__ == "__main__":
    main()