
import torch
from PIL import Image
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration

def setup_llava():
    # Load the model and processor
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, processor

def process_images(image_folder, model, processor):
    # List to store results
    results = []
    
    # Get all image files from the folder
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Load and process image
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        
        # Prepare prompt - you can customize this
        prompt = "What do you see in this image?"
        
        # Process image and text
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device="cuda", dtype=torch.float16)
        
        # Generate response
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode response
        response = processor.decode(output[0], skip_special_tokens=True)
        results.append({
            'image_file': image_file,
            'response': response
        })
    
    return results

def main():
    # Set your image folder path
    image_folder = "path/to/your/images"
    
    # Initialize model and processor
    model, processor = setup_llava()
    
    try:
        # Process all images in the folder
        results = process_images(image_folder, model, processor)
        
        # Print results
        for result in results:
            print(f"\nImage: {result['image_file']}")
            print(f"Description: {result['response']}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()