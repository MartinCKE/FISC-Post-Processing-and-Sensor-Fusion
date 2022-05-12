# 81358_FISC_Post_Processing

Post processing and AI-related tasks on FISC data.


Installation guide.

1. Download repository 
2. Download and install anaconda or miniconda which matches the Python version you’ve got (> 3.7). Anaconda is required to create a virtual environment with the specific version of all packages to run the software.
3. After conda is installed, create a virtual environment by using commands:
    1. If no GPU acceleration is used, run: “conda env create -f conda-cpu.yml”
    2. If CUDA-supported GPU is used, run: “conda env create -f conda-cgpu.yml”. This requires CUDA Toolkit 10.1 to be installed (https://developer.nvidia.com/cuda-10.1-download-archive-update2)
4. Hopefully, no errors occur here. If so. run: “conda activate yolov4-cpu” or “conda activate yolov4-gpu”.
5. Only 3 videos are included in the repository due to file size constraints.
   All videos from the second field test can be downloaded from:
   https://drive.google.com/file/d/1oTzxz269bf6eGfWYmWoPtyDgbs2yqzVT/view?usp=sharing
6. If you are going to run the YOLOv4-DeepSORT software (generate), you need the custom trained weight files or the YOLOv4 model. 
   These are downloaded from: https://drive.google.com/file/d/1AZcUAWdVLyUQrFmTyyS0RiiCWRj2nww7/view?usp=sharing

Place these videos in the data/cam_recordings/secondTest/compress directory.
Now, you're ready to test the FISC processing software! 

NOTE: No videos from the initial field test are included in the repository, so any attempts to visualize acoustic AND visual data without the "--ace" argument will not do anything.


Below are some useful examples of function calls to generate certain data:

python3 viewSavedData.py --imu --ace
    - Generates directional vertical inclination plot over time from second field test. Remove "--ace" to show same data from initial field test.

python3 viewSavedData.py --o2temp --profile --ace
    - Generates environmental parameter score based on custom optimal functions as function of depth (depth-profile). Also visualizes aquatic score heat map and sound velocity profile.


python3 viewSavedData.py --syncPlot --sf --show --ace --savePlots --start 13:05
	- Generates time-synchronized video frame and acoustic ping from second field test, starting for 13:05 o'clock. Exports a figure with both optical and acoustic sample in same image.


python3 fusion.py --generate
	- Runs the YOLOv4-DeepSORT implementation to generate CSV files containing tracker information on every individual on every timestamp coinciding with acoustic ping. Requires conda environment. Follow guide above to set up the environment. (Not necessary since pre-generated files are included in repository)

python3 fusion.py --p --ID 145 --start 13:07 --stop 13:08 --savePlots
	- Analyzes acoustic data and the pre-generated tracker files to perform sensor fusion for size and swimming speed estimation. This specific call shows a sequence with a good fusion match on ID 145.





