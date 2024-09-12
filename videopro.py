import os
from tkinter import filedialog
from video_prp import *
def vidpro():
	file_paths = filedialog.askopenfilenames(title="Select Videos", filetypes=[("Video files", "*.mp4;*.avi")])
	if file_paths:
		for path in file_paths:
			video_check(path)