echo "Starting ratio generation..."
python ratio_generation.py
echo "Starting training"
python training.py
echo "Generating Figures"
python generate_figures.py
