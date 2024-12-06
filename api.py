from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering
import uvicorn
from typing import Optional

# Initialize FastAPI with metadata
app = FastAPI(
    title="Visual Question Answering API",
    description="API for answering questions about images using VILT model",
    version="0.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

cache_dir = "./model_cache"


def load_model():
    try:
        processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa",
            cache_dir=cache_dir
        )
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa",
            cache_dir=cache_dir
        )
        return processor, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


# Load model at startup
processor, model = load_model()


def validate_image(image_bytes: bytes) -> Image.Image:
    """Validate and convert uploaded image bytes to PIL Image."""
    try:
        image = Image.open(BytesIO(image_bytes))
        if image.format not in ['JPEG', 'PNG']:
            raise ValueError("Unsupported image format")
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def get_answer(image: Image.Image, text: str) -> str:
    try:
        # Prepare input
        encoding = processor(image, text, return_tensors="pt")

        # Forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return answer
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")


@app.get("/", include_in_schema=False)
async def index():
    """Redirect root to documentation."""
    return RedirectResponse(url="/docs")


@app.post("/answer",
          response_model=dict,
          summary="Get answer for image-based question",
          response_description="Returns the answer to the question about the image"
          )
async def process_image(
        image: UploadFile = File(..., description="Image file (JPEG/PNG)"),
        question: str = Query(...,
                              description="Question about the image",
                              min_length=1,
                              max_length=200)
):
    """
    Process an image and question to generate an answer.

    Args:
        image: Image file (JPEG or PNG)
        question: Question about the image

    Returns:
        JSON response with answer or error message
    """
    try:
        # Validate content type
        if not image.content_type in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are supported"
            )

        # Read and validate image
        image_bytes = await image.read()
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="Image file too large (max 10MB)"
            )

        # Process image and get answer
        img = validate_image(image_bytes)
        answer = get_answer(img, question)

        return {
            "success": True,
            "answer": answer,
            "question": question
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Add health check endpoint
@app.get("/health")
async def health_check():
    """Check if the service is healthy."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)