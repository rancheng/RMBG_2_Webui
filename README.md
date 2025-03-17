# Background Removal Tool

This is a simple web application built with Gradio that uses the RMBG-2.0 model to remove backgrounds from images. The tool can process images uploaded directly or via URL, and outputs both the transparent result and the generated mask.

## Features

- Upload images or provide image URLs
- Real-time background removal using AI
- Download transparent PNG images without backgrounds
- Download the generated mask
- GPU acceleration (if available)
- Automatic dependency installation

## Installation

### Using Conda (Recommended)

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. Create a conda environment using the provided YAML file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate rmbg
   ```

### Manual Installation

1. Create a new conda environment:
   ```bash
   conda create -n rmbg python=3.10
   conda activate rmbg
   ```

2. Install the required packages:
   ```bash
   pip install gradio torch torchvision transformers timm pillow requests
   ```

## Usage

1. Run the application:
   ```bash
   python bg_remove_app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:7860`

3. Upload an image using the file picker or paste an image URL

4. Click "Remove Background" to process the image

5. The transparent result and mask will appear on the right side

6. Right-click on either image to download

## Dependencies

- Python 3.10+
- timm
- transformers
- torch
- torchvision
- gradio
- Pillow
- requests

## Troubleshooting

If you encounter any errors:

1. Make sure all dependencies are installed:
   ```bash
   pip install timm transformers torch torchvision gradio pillow requests
   ```

2. Check if your GPU is properly configured (for faster processing)

3. If you're getting CUDA errors, try running on CPU by modifying the code

## Example

![Example of background removal](example.png)

## License

This project uses the RMBG-2.0 model from briaai. Please check their license for model usage terms. 