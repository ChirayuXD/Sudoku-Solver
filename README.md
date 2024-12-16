# Sudoku Solver with Computer Vision and Backtracking

## Overview

This project is an advanced Sudoku solver that uses computer vision techniques to detect, extract, and solve Sudoku puzzles from images. The application combines image processing, optical character recognition (OCR), and a backtracking algorithm to automatically solve Sudoku puzzles.

## Features

- **Image Preprocessing**: Converts input images to a clean, high-contrast format suitable for digit recognition
- **Grid Detection**: Identifies and extracts the Sudoku grid from the input image
- **Digit Recognition**: Uses Tesseract OCR to recognize digits in the grid
- **Backtracking Solver**: Implements an efficient algorithm to solve the Sudoku puzzle
- **Solution Visualization**: Overlays the solution back onto the original image

## Prerequisites

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Pytesseract
- Tesseract OCR installed on your system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sudoku-solver.git
cd sudoku-solver
```

2. Install required dependencies:
```bash
pip install opencv-python numpy pytesseract
```

3. Install Tesseract OCR:
   - Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

## Configuration

- Update the Tesseract path in the script:
```python
pytesseract.pytesseract.tesseract_cmd = r"PATH_TO_TESSERACT"
```

- Set the input image path in `main()`:
```python
image_path = "path/to/your/sudoku_image.jpg"
```

## Usage

Run the script with your Sudoku image:
```bash
python sudoku_solver.py
```

## How It Works

1. **Preprocess Image**: 
   - Resize image
   - Convert to grayscale
   - Apply adaptive thresholding
   - Perform morphological operations

2. **Grid Detection**:
   - Find grid contours
   - Detect and order grid corners
   - Apply perspective transform

3. **Digit Extraction**:
   - Divide grid into 9x9 cells
   - Use Tesseract OCR to recognize digits
   - Create initial Sudoku board

4. **Solve Puzzle**:
   - Validate input board
   - Use backtracking algorithm to solve
   - Overlay solution on original image

## Output

The script generates:
- Intermediate processing images in `sudoku_output_images/`
- Final solution overlay image
- Console output with solving time

## Performance

- Utilizes efficient backtracking algorithm
- Measures and displays solving runtime

## Limitations

- Requires clear, high-contrast Sudoku images
- Works best with printed/digital Sudoku puzzles
- Accuracy depends on image quality and OCR performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here, e.g., MIT License]

## Acknowledgments

- OpenCV
- Pytesseract
- Tesseract OCR Project
