from transformers.pipelines import pipeline
from transformers import AutoImageProcessor
from PIL import Image
import torch
import os
from datetime import datetime
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END, START
from langfuse.callback import CallbackHandler

# Set torch precision
torch.set_float32_matmul_precision("high")

# Initialize Langfuse for observability (optional)
# Uncomment and configure if you want to use Langfuse
from langfuse import Langfuse
LANGFUSE_PUBLIC_KEY = "pk-lf-e0a91ba6-fe2f-43a0-930f-d391ed91be4d"
LANGFUSE_SECRET_KEY = "sk-lf-10993a05-3b42-4780-a055-aa039d953273"
LANGFUSE_HOST = "https://cloud.langfuse.com"
langfuse = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)

# Define the state structure for our workflow
class MedicalAnalysisState(TypedDict):
    image_path: str
    image: Optional[Image.Image]
    image_loaded: bool
    analysis_result: str
    errors: List[str]
    timestamp: str

def initialize_models():
    """Initialize MedGemma model"""
    print("Initializing MedGemma model...")
    processor = AutoImageProcessor.from_pretrained(
        "google/medgemma-4b-it",
        use_fast=True
        )
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device="cpu",
        image_processor=processor
    )
    
    return pipe

# LangGraph node functions
def load_image(state: MedicalAnalysisState) -> MedicalAnalysisState:
    """Load and validate the image"""
    try:
        print(f"Loading image: {state['image_path']}")
        image = Image.open(state['image_path'])
        
        return {
            **state,
            "image": image,
            "image_loaded": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        error_msg = f"Failed to load image: {str(e)}"
        print(error_msg)
        
        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "image_loaded": False
        }

def analyze_knee_image(state: MedicalAnalysisState, pipe) -> MedicalAnalysisState:
    """Analyze the knee ultrasound image using MedGemma"""
    if not state["image_loaded"]:
        return state
    
    try:
        print("Analyzing image with MedGemma...")
        
        # Prepare messages for MedGemma
        inflammation_knee = True  # You can modify this based on your needs
        if inflammation_knee:
            first_message = "This person has inflammation in their knee based on the ultrasound"
        else:
            first_message = "This person does not have inflammation in their knee based on the ultrasound"

        second_message = (
            "On the left is the original ultrasound image. On the right is the segmented region of interest."
            "Based on the ultrasound knee image, please describe the affected anatomical structures and any visible abnormalities. "
            "Describe findings such as joint effusion, synovial thickening, bone spurs, or cartilage erosion. "
            "Then, based on these findings, identify the most likely diagnosis among the following: "
            "Baker's cyst, osteophytes (bone spurs), cartilage degeneration, joint effusion, or synovitis. "
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
                    {"type": "image", "image": state["image"]},
                ]
            }
        ]

        # Generate analysis
        output = pipe(text=messages, max_new_tokens=200)
        analysis_result = output[0]["generated_text"][-1]["content"]
        
        return {
            **state,
            "analysis_result": analysis_result
        }
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(error_msg)
        
        return {
            **state,
            "errors": state["errors"] + [error_msg]
        }

def save_results(state: MedicalAnalysisState) -> MedicalAnalysisState:
    """Save analysis results to file"""
    try:
        filename = "analysis_results.txt"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {state['timestamp']}\n")
            f.write(f"Image Path: {state['image_path']}\n")
            f.write(f"\nAnalysis Result:\n{state['analysis_result']}\n")
            if state['errors']:
                f.write(f"\nErrors:\n" + "\n".join(state['errors']) + "\n")
            f.write(f"{'='*80}\n")
        
        print(f"Results saved to {filename}")
        return state
    except Exception as e:
        error_msg = f"Failed to save results: {str(e)}"
        print(error_msg)
        return {
            **state,
            "errors": state["errors"] + [error_msg]
        }

def check_for_errors(state: MedicalAnalysisState) -> str:
    """Route based on whether there are errors"""
    if state["errors"]:
        return "error_handler"
    return "save_results"

def error_handler(state: MedicalAnalysisState) -> MedicalAnalysisState:
    """Handle errors in the workflow"""
    print("Workflow encountered errors:")
    for error in state["errors"]:
        print(f"  - {error}")
    return state

def create_workflow(pipe):
    """Create the LangGraph workflow"""
    # Create the state graph
    workflow = StateGraph(MedicalAnalysisState)
    
    # Add nodes
    workflow.add_node("load_image", load_image)
    workflow.add_node("analyze_knee_image", lambda state: analyze_knee_image(state, pipe))
    workflow.add_node("save_results", save_results)
    workflow.add_node("error_handler", error_handler)
    
    # Set entry point
    workflow.set_entry_point("load_image")
    
    # Add edges
    workflow.add_edge("load_image", "analyze_knee_image")
    workflow.add_edge("save_results", END)
    workflow.add_edge("error_handler", END)
    
    # Add conditional edge from analyze_knee_image
    workflow.add_conditional_edges(
        "analyze_knee_image",
        check_for_errors,
        {
            "error_handler": "error_handler",
            "save_results": "save_results"
        }
    )
    
    # Compile the graph
    return workflow.compile()

def main():
    """Main function to run the analysis workflow"""
    # Initialize the model
    pipe = initialize_models()
    print("Model initialized successfully!")
    
    # Create the workflow
    app = create_workflow(pipe)
    print("Workflow created successfully!")
    
    # Main loop
    while True:
        print("\n" + "="*50)
        print("Knee Ultrasound Analysis Tool (LangGraph)")
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
        
        # Initialize state
        initial_state = MedicalAnalysisState(
            image_path=image_path,
            image=None,
            image_loaded=False,
            analysis_result="",
            errors=[],
            timestamp=""
        )
        
        # Run the workflow
        try:
            print("Starting analysis workflow...")
            
            # Create Langfuse callback handler for tracing
            langfuse_handler = CallbackHandler(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            
            # Run workflow with Langfuse tracing
            result = app.invoke(
                initial_state,
                config={"callbacks": [langfuse_handler]}
            )
            
            # Display results
            print("\n" + "="*50)
            print("ANALYSIS RESULTS")
            print("="*50)
            print(f"\nAnalysis Result:\n{result['analysis_result']}")
            
            if result['errors']:
                print(f"\nErrors encountered:\n" + "\n".join(result['errors']))
                
        except Exception as e:
            print(f"Workflow execution failed: {e}")

if __name__ == "__main__":
    main() 