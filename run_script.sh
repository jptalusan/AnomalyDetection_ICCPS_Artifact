echo "Starting ratio generation..."
python ratio_generation.py
echo "Starting training, might take +350s"
python training.py
echo "Generating Figures (untested with latest dataset)"
python generate_figures.py
