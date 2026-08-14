from dataclasses import dataclass


class HandwritingOCRUnavailable(Exception):
    pass


@dataclass
class OCRCellResult:
    text: str
    confidence: float


class TrOCRHandwritingRecognizer:
    _processor = None
    _model = None

    def __init__(self):
        if self.__class__._processor is None or self.__class__._model is None:
            try:
                from transformers import RobertaTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel
            except ImportError as exc:
                raise HandwritingOCRUnavailable(
                    "Handwriting OCR is not installed. Install transformers, torch, and the microsoft/trocr-base-handwritten model."
                ) from exc

            try:
                try:
                    self.__class__._processor = TrOCRProcessor.from_pretrained(
                        "microsoft/trocr-base-handwritten",
                        use_fast=False,
                    )
                except Exception:
                    image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
                    self.__class__._processor = TrOCRProcessor(image_processor, tokenizer)
                self.__class__._model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            except Exception as exc:
                raise HandwritingOCRUnavailable(
                    "Handwriting OCR model is not available. Download microsoft/trocr-base-handwritten before scanning."
                ) from exc

        self.processor = self.__class__._processor
        self.model = self.__class__._model

    def read(self, image):
        pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        confidence = 0.86 if text else 0.0
        return OCRCellResult(text=text, confidence=confidence)
