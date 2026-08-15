from dataclasses import dataclass
import hashlib
import threading
import time


class HandwritingOCRUnavailable(Exception):
    pass


@dataclass
class OCRCellResult:
    text: str
    confidence: float


class TrOCRHandwritingRecognizer:
    _processor = None
    _model = None
    _torch = None
    _load_lock = threading.Lock()
    _inference_count = 0
    _cache = {}

    def __init__(self):
        self.__class__.load_once()

        self.processor = self.__class__._processor
        self.model = self.__class__._model
        self.torch = self.__class__._torch

    @classmethod
    def load_once(cls):
        if cls._processor is not None and cls._model is not None:
            print("[OCR] TrOCR model already loaded")
            return

        with cls._load_lock:
            if cls._processor is not None and cls._model is not None:
                print("[OCR] TrOCR model already loaded")
                return

            print("[OCR] START TrOCR model initialization")
            start = time.perf_counter()
            try:
                import torch
                from transformers import RobertaTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel
            except ImportError as exc:
                raise HandwritingOCRUnavailable(
                    "Handwriting OCR is not installed. Install transformers, torch, and the microsoft/trocr-base-handwritten model."
                ) from exc

            try:
                try:
                    cls._processor = TrOCRProcessor.from_pretrained(
                        "microsoft/trocr-base-handwritten",
                        use_fast=False,
                        local_files_only=True,
                    )
                except Exception:
                    image_processor = ViTImageProcessor.from_pretrained(
                        "microsoft/trocr-base-handwritten",
                        local_files_only=True,
                    )
                    tokenizer = RobertaTokenizer.from_pretrained("roberta-large", local_files_only=True)
                    cls._processor = TrOCRProcessor(image_processor, tokenizer)
                cls._model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-base-handwritten",
                    local_files_only=True,
                )
                cls._model.eval()
                cls._torch = torch
            except Exception as exc:
                raise HandwritingOCRUnavailable(
                    "Handwriting OCR model is not available locally. Download microsoft/trocr-base-handwritten before scanning."
                ) from exc
            finally:
                print(f"[OCR] END TrOCR model initialization: {time.perf_counter() - start:.2f}s")

    @classmethod
    def reset_counter(cls):
        cls._inference_count = 0

    @classmethod
    def inference_count(cls):
        return cls._inference_count

    def read(self, image, max_new_tokens=16, cache_key=None):
        if cache_key and cache_key in self.__class__._cache:
            return self.__class__._cache[cache_key]
        if cache_key is None:
            cache_key = self.cache_key(image, max_new_tokens)
            if cache_key in self.__class__._cache:
                return self.__class__._cache[cache_key]

        result = self.read_many([image], max_new_tokens=max_new_tokens)[0]
        self.__class__._cache[cache_key] = result
        return result

    def read_batch(self, items):
        """Run TrOCR on several images with a single inference call per token group.

        items: list of (cache_key, image, max_new_tokens). Returns a dict of
        cache_key -> OCRCellResult. Cached results are reused, and the remaining
        images are batched into as few model.generate calls as possible.
        """
        results = {}
        pending = {}
        for key, image, max_new_tokens in items:
            if key in self.__class__._cache:
                results[key] = self.__class__._cache[key]
            else:
                pending.setdefault(max_new_tokens, []).append((key, image))

        for max_new_tokens, entries in pending.items():
            keys = [key for key, _ in entries]
            images = [image for _, image in entries]
            batch_results = self.read_many(images, max_new_tokens=max_new_tokens)
            for key, result in zip(keys, batch_results):
                self.__class__._cache[key] = result
                results[key] = result
        return results

    @staticmethod
    def cache_key(image, max_new_tokens):
        rgb = image.convert("RGB")
        digest = hashlib.sha1(rgb.tobytes()).hexdigest()
        return f"{rgb.size}:{max_new_tokens}:{digest}"

    def read_many(self, images, max_new_tokens=16):
        self.__class__._inference_count += 1
        count = self.__class__._inference_count
        print(f"[OCR] TrOCR inference #{count}")
        print(f"[OCR] START TrOCR inference #{count}")
        start = time.perf_counter()
        rgb_images = [image.convert("RGB") for image in images]
        pixel_values = self.processor(images=rgb_images, return_tensors="pt").pixel_values
        import torch.nn.functional as functional

        with self.torch.inference_mode():
            outputs = self.model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
        sequences = outputs.sequences
        texts = [text.strip() for text in self.processor.batch_decode(sequences, skip_special_tokens=True)]

        # Per-sequence confidence: mean softmax probability of the generated tokens.
        confidences = []
        if outputs.scores:
            log_probs = []
            for step, score in enumerate(outputs.scores):
                chosen = sequences[:, 1 + step]
                log_probs.append(
                    functional.log_softmax(score, dim=-1)
                    .gather(1, chosen.unsqueeze(1))
                    .squeeze(1)
                )
            stacked = self.torch.stack(log_probs, dim=1)
            confidences = [
                float(self.torch.exp(stacked[i].mean()).item())
                for i in range(stacked.shape[0])
            ]
        else:
            confidences = [0.86 if text else 0.0 for text in texts]
        print(f"[OCR] END TrOCR inference #{count}: {time.perf_counter() - start:.2f}s")
        return [
            OCRCellResult(text=text, confidence=round(conf, 3) if text else 0.0)
            for text, conf in zip(texts, confidences)
        ]
