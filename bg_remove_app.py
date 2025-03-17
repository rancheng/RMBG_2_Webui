import io
import requests
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
import subprocess
import sys
import importlib.util

# Check and install required dependencies
def check_and_install_dependencies():
    required_packages = ['timm', 'transformers', 'torch', 'torchvision']
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Dependencies installed successfully!")
            # Reload modules if needed
            if 'transformers' in missing_packages:
                importlib.reload(importlib.import_module('transformers'))
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install dependencies. Please run: pip install {' '.join(missing_packages)}")
            return False
    return True

# Now import the module that requires timm after checking dependencies
if check_and_install_dependencies():
    from transformers import AutoModelForImageSegmentation
else:
    print("Error: Required dependencies missing. The app may not function correctly.")

# Load the model
def load_model():
    try:
        model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        if torch.cuda.is_available():
            model = model.to('cuda')
        model.eval()
        return model, None
    except ImportError as e:
        return None, f"Error loading model: {str(e)}. Please install missing dependencies with 'pip install timm'."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Data transform settings
def get_transforms(image_size=(1024, 1024)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Function to process images
def process_image(image, model, transform):
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Transform the image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Get mask
    pred = preds[0].squeeze()
    mask_image = transforms.ToPILImage()(pred)
    mask_resized = mask_image.resize(image.size)
    
    # Apply mask to original image
    result_image = image.copy()
    result_image.putalpha(mask_resized)
    
    return result_image, mask_resized

# Function to fetch image from URL
def fetch_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        return None

# Main function for handling input
def remove_background(input_image, url):
    # Initialize model and transform
    model, error = load_model()
    if error:
        return None, None, error
    
    transform = get_transforms()
    
    # Process from URL if provided
    if url and not input_image:
        image = fetch_image_from_url(url)
        if image is None:
            return None, None, "Failed to fetch image from URL"
    # Otherwise use uploaded image
    elif input_image is not None:
        image = Image.open(input_image).convert("RGB")
    else:
        return None, None, "Please provide an image or URL"
    
    # Process the image
    try:
        transparent_result, mask = process_image(image, model, transform)
        return transparent_result, mask, "Processing successful"
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Background Removal Tool") as demo:
    gr.Markdown("# Background Removal Tool")
    gr.Markdown("Upload an image or enter an image URL to remove the background")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Upload Image")
            url_input = gr.Textbox(label="Or enter image URL")
            process_btn = gr.Button("Remove Background")
        
        with gr.Column():
            transparent_output = gr.Image(label="Transparent Result", type="pil")
            mask_output = gr.Image(label="Mask", type="pil")
            status_text = gr.Textbox(label="Status")
    
    process_btn.click(
        fn=remove_background,
        inputs=[input_image, url_input],
        outputs=[transparent_output, mask_output, status_text]
    )
    
    gr.Markdown("### Instructions")
    gr.Markdown("1. Upload an image using the file picker or paste a URL to an image")
    gr.Markdown("2. Click 'Remove Background' to process the image")
    gr.Markdown("3. The transparent result and mask will appear on the right")
    gr.Markdown("4. Right-click on either image to download")
    gr.Markdown("### Required Dependencies")
    gr.Markdown("This app requires the following packages: `timm`, `transformers`, `torch`, `torchvision`")
    gr.Markdown("If you encounter errors, install missing dependencies with: `pip install timm transformers torch torchvision`")

# Launch the app
if __name__ == "__main__":
    demo.launch() 