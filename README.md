# 81358_FISC_Post_Processing

Post processing and AI-related tasks on FISC data.

Below are some useful examples of function calls to generate certain data:

python3 viewSavedData.py --syncPlot --sf --show --ace --savePlots --start 13:05
	- Generates time-synchronized video frame and acoustic ping from second field test, starting for 13:05 o'clock. Exports a figure with both optical and acoustic sample in same image.


python3 fusion.py --generate
	- Runs the YOLOv4-DeepSORT implementation to generate CSV files containing tracker information on every individual on every timestamp coinciding with acoustic ping. Requires conda environment. Follow guide above to set up the environment. (Not necessary since pre-generated files are included in repository)

python3 fusion.py --p --ID 145 --start 13:07 --stop 13:08 --savePlots
	- Analyzes acoustic data and the pre-generated tracker files to perform sensor fusion for size and swimming speed estimation. This specific call shows a sequence with a good fusion match on ID 145.





