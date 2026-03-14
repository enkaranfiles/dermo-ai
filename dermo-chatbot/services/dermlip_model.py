"""DermLIP (CLIP-based) visual classifier for skin conditions."""

import open_clip
import torch
from PIL import Image

from services.conditions import SKIN_CONDITIONS, PROMPT_TEMPLATE


class DermLIPClassifier:
    """Zero-shot skin condition classifier using DermLIP ViT-B/16."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "hf-hub:redlessone/DermLIP_ViT-B-16"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.model.eval().to(self.device)

        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Pre-compute text embeddings for all conditions (they never change)
        text_inputs = self.tokenizer(
            [PROMPT_TEMPLATE.format(condition=c) for c in SKIN_CONDITIONS]
        )
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs.to(self.device))
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """Return top-k predictions as [{condition, confidence}, ...]."""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

        scores, indices = similarity[0].topk(top_k)
        return [
            {"condition": SKIN_CONDITIONS[idx], "confidence": round(score.item(), 4)}
            for score, idx in zip(scores, indices)
        ]
