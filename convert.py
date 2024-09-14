import os
from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            # Full file path for input and output
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, filename.replace(".mp4", ".gif"))
            
            # Load the video and convert to gif
            clip = VideoFileClip(input_path)
            clip.write_gif(output_path)
            print(f"Converted {filename} to GIF")

# Replace '/path/to/folder' with the actual folder path containing mp4 files
folder_path = '/mnt/lustre/sichenyang.p/code/SD3_Vid/Vchitect-2.0/assets/samples'
convert_mp4_to_gif(folder_path)