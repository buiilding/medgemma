from transformers import pipeline, AutoImageProcessor
from PIL import Image
import torch
import google.generativeai as genai
import os
from datetime import datetime

# Set torch precision
torch.set_float32_matmul_precision("high")

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyB5bF8Ch9Dqu6t_6G7TKgQzlaRKcJSSzD8"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

def initialize_models():
    """Initialize both MedGemma and Gemini models"""
    print("Initializing MedGemma model...")
    processor = AutoImageProcessor.from_pretrained(
        "google/medgemma-4b-it",  # or whatever checkpoint you use
        use_fast=True
        )
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device="cuda",
        image_processor=processor
    )
    
    print("Initializing Gemini model...")
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    
    return pipe, gemini_model

def translate_to_vietnamese(text, gemini_model):
    """Translate text to Vietnamese using Gemini"""
    try:
        prompt = f"Translate the following medical text to Vietnamese. Keep medical terms accurate and professional, only return the translated text:\n\n{text}"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def analyze_knee_image(image_path, pipe, gemini_model):
    """Analyze knee ultrasound image and translate to Vietnamese"""
    try:
        # Load image
        image = Image.open(image_path)
        print(f"Successfully loaded image: {image_path}")
        
        # Prepare messages for MedGemma
        inflammation_knee = True  # You can modify this based on your needs
        if inflammation_knee:
            first_message = "This person has inflammation in their knee based on the ultrasound"
        else:
            first_message = "This person does not have inflammation in their knee based on the ultrasound"

        second_message = (
            "Based on the ultrasound knee image, please describe the affected anatomical structures and any visible abnormalities. "
            "Describe findings such as joint effusion, synovial thickening, bone spurs, or cartilage erosion. "
            "Then, based on these findings, identify the most likely diagnosis among the following: "
            "Baker’s cyst, osteophytes (bone spurs), cartilage degeneration, joint effusion, or synovitis. "
            "Your response should follow this format:\n\n"
            "'The condition affects the [bone/region]. [Finding] is seen in the [specific location], suggesting [diagnosis].'"
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_message},
                    {"type": "text", "text": second_message},
                    {"type": "image", "image": image},
                ]
            }
        ]

        # Generate analysis
        print("Generating analysis...")
        output = pipe(text=messages, max_new_tokens=200)
        english_analysis = output[0]["generated_text"][-1]["content"]
        
        # Translate to Vietnamese
        print("Translating to Vietnamese...")
        vietnamese_analysis = translate_to_vietnamese(english_analysis, gemini_model)
        
        return english_analysis, vietnamese_analysis
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None, None

def save_to_file(image_path, english_text, vietnamese_text, filename="analysis_results.txt"):
    """Save analysis results to a text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Image Path: {image_path}\n")
        f.write(f"\nEnglish Analysis:\n{english_text}\n")
        f.write(f"\nVietnamese Translation:\n{vietnamese_text}\n")
        f.write(f"{'='*80}\n")

def main():
    """Main function to run the analysis loop"""
    # Check if Gemini API key is set
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("Please set your Gemini API key in the GEMINI_API_KEY variable!")
        return
    
    # Initialize models
    pipe, gemini_model = initialize_models()
    print("Models initialized successfully!")
    
    # Main loop
    while True:
        print("\n" + "="*50)
        print("Knee Ultrasound Analysis Tool")
        print("="*50)
        
        # Get image path from user
        image_path = input("Enter the path to the knee ultrasound image (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: File not found at {image_path}")
            continue
        
        # Analyze image
        english_result, vietnamese_result = analyze_knee_image(image_path, pipe, gemini_model)
        
        if english_result and vietnamese_result:
            # Print results
            print("\n" + "="*50)
            print("ANALYSIS RESULTS")
            print("="*50)
            print(f"\nEnglish Analysis:\n{english_result}")
            print(f"\nVietnamese Translation:\n{vietnamese_result}")
            
            # Save to file
            save_to_file(image_path, english_result, vietnamese_result)
            print(f"\nResults saved to analysis_results.txt")
        else:
            print("Analysis failed. Please try again with a different image.")

if __name__ == "__main__":
    main() 