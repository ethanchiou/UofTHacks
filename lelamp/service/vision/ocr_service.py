"""
OCR Service - Extract text from images using Tesseract OCR.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Union, List
from dataclasses import dataclass
from pathlib import Path

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None


@dataclass
class OCRResult:
    """Result from OCR text extraction"""
    text: str                    # Extracted text (cleaned)
    raw_text: str               # Raw text with all whitespace
    confidence: float           # Average confidence (0-100)
    word_count: int             # Number of words detected
    lines: List[str]            # Text split by lines
    boxes: List[dict] = None    # Bounding boxes for each word (optional)


class OCRService:
    """
    Optical Character Recognition service.
    Extracts text from images using Tesseract OCR.
    
    Usage:
        ocr = OCRService()
        
        # From file
        result = ocr.extract_text("photo.jpg")
        print(result.text)
        
        # From numpy array (OpenCV frame)
        result = ocr.extract_text(frame)
        print(result.text)
        
        # Get word bounding boxes
        result = ocr.extract_text(frame, get_boxes=True)
        for box in result.boxes:
            print(f"Word: {box['text']} at ({box['x']}, {box['y']})")
    """
    
    def __init__(self, lang: str = "eng", tesseract_cmd: str = None):
        """
        Initialize OCR service.
        
        Args:
            lang: Language code for Tesseract (default: "eng")
                  Common: "eng", "fra", "deu", "spa", "chi_sim", "jpn"
            tesseract_cmd: Path to tesseract executable (auto-detected if None)
        """
        self.logger = logging.getLogger("service.OCRService")
        self.lang = lang
        
        if not TESSERACT_AVAILABLE:
            self.logger.error("pytesseract not installed. Run: pip install pytesseract")
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Test tesseract availability
        try:
            pytesseract.get_tesseract_version()
            self.logger.info(f"OCR Service initialized (lang={lang})")
        except Exception as e:
            self.logger.error(f"Tesseract not found. Install with: sudo apt install tesseract-ocr")
            raise RuntimeError(f"Tesseract OCR not available: {e}")
    
    def extract_text(
        self, 
        image: Union[str, Path, np.ndarray, bytes],
        get_boxes: bool = False,
        preprocess: bool = True,
        config: str = ""
    ) -> OCRResult:
        """
        Extract text from an image.
        
        Args:
            image: Image source - file path, numpy array, or bytes
            get_boxes: If True, include bounding boxes for each word
            preprocess: If True, apply preprocessing for better accuracy
            config: Additional Tesseract config string
            
        Returns:
            OCRResult with extracted text and metadata
        """
        # Load image
        img = self._load_image(image)
        
        if img is None:
            return OCRResult(
                text="",
                raw_text="",
                confidence=0.0,
                word_count=0,
                lines=[],
                boxes=[]
            )
        
        # Preprocess for better OCR
        if preprocess:
            img = self._preprocess(img)
        
        # Build config
        ocr_config = f"-l {self.lang}"
        if config:
            ocr_config += f" {config}"
        
        # Extract text with confidence data
        try:
            data = pytesseract.image_to_data(
                img, 
                config=ocr_config,
                output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return OCRResult(
                text="",
                raw_text="",
                confidence=0.0,
                word_count=0,
                lines=[],
                boxes=[]
            )
        
        # Process results
        words = []
        confidences = []
        boxes = [] if get_boxes else None
        
        for i, text in enumerate(data['text']):
            text = text.strip()
            if text:
                words.append(text)
                conf = int(data['conf'][i])
                if conf > 0:  # -1 means no confidence
                    confidences.append(conf)
                
                if get_boxes:
                    boxes.append({
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': conf,
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i]
                    })
        
        # Build text output
        raw_text = pytesseract.image_to_string(img, config=ocr_config)
        clean_text = ' '.join(words)
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRResult(
            text=clean_text,
            raw_text=raw_text,
            confidence=avg_confidence,
            word_count=len(words),
            lines=lines,
            boxes=boxes
        )
    
    def extract_text_simple(self, image: Union[str, Path, np.ndarray, bytes]) -> str:
        """
        Simple method - just returns the extracted text string.
        
        Args:
            image: Image source
            
        Returns:
            Extracted text as string
        """
        result = self.extract_text(image, get_boxes=False, preprocess=True)
        return result.text
    
    def _load_image(self, image: Union[str, Path, np.ndarray, bytes]) -> Optional[np.ndarray]:
        """Load image from various sources."""
        try:
            if isinstance(image, np.ndarray):
                return image
            
            if isinstance(image, (str, Path)):
                path = str(image)
                img = cv2.imread(path)
                if img is None:
                    self.logger.error(f"Could not load image: {path}")
                return img
            
            if isinstance(image, bytes):
                nparr = np.frombuffer(image, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            self.logger.error(f"Unsupported image type: {type(image)}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        """
        # Convert to grayscale if color
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply adaptive thresholding for better text contrast
        # This helps with varying lighting conditions
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Optional: Denoise
        processed = cv2.medianBlur(processed, 3)
        
        return processed
    
    def draw_boxes(self, image: np.ndarray, result: OCRResult) -> np.ndarray:
        """
        Draw bounding boxes and text on image for visualization.
        
        Args:
            image: Original image (numpy array)
            result: OCRResult with boxes
            
        Returns:
            Image with boxes drawn
        """
        if not result.boxes:
            return image
        
        output = image.copy()
        
        for box in result.boxes:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            conf = box['confidence']
            
            # Color based on confidence (green=high, red=low)
            color = (0, int(conf * 2.55), int((100 - conf) * 2.55))
            
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output, box['text'], (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output


# Convenience function for quick use
def extract_text(image: Union[str, Path, np.ndarray, bytes], lang: str = "eng") -> str:
    """
    Quick function to extract text from an image.
    
    Args:
        image: Image file path, numpy array, or bytes
        lang: Language code (default: "eng")
        
    Returns:
        Extracted text string
    """
    ocr = OCRService(lang=lang)
    return ocr.extract_text_simple(image)


if __name__ == "__main__":
    # Test the OCR service
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_service.py <image_path>")
        print("Example: python ocr_service.py photo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Extracting text from: {image_path}")
    print("-" * 40)
    
    try:
        ocr = OCRService()
        result = ocr.extract_text(image_path, get_boxes=True)
        
        print(f"Text found ({result.word_count} words, {result.confidence:.1f}% confidence):")
        print("-" * 40)
        print(result.text)
        print("-" * 40)
        
        if result.lines:
            print(f"\nLines ({len(result.lines)}):")
            for i, line in enumerate(result.lines, 1):
                print(f"  {i}: {line}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
