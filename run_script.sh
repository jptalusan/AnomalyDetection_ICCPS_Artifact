echo "Starting ratio generation..."
python ratio_generation.py
echo "Starting training, might take +350s"
python training.py
echo "Generating Figures (unfinished)"
python generate_figures.py
