"Extract features from images"

import torch


class FeatureExtractor:
    "Extract features from images"

    def __init__(self, model, processor, has_cls, num_register):
        self.model = model
        self.processor = processor
        self.has_cls = has_cls
        self.num_register = num_register

    def extract_features(self, image):
        "Extracts features of a image"
        processed_img = self.processor(images=image, return_tensors="pt").to(
            self.model.device
        )
        with torch.inference_mode():
            features = self.model(**processed_img).last_hidden_state

        return (
            {}
            | self.extract_cls_token(features)
            | self.extract_register_tokens(features)
            | self.extract_patch_tokens(features)
        )

    def extract_cls_token(self, features):
        "Extract the CLS token from the features"
        if self.has_cls:
            return {"cls": features[0, 0]}
        return {}

    def extract_register_tokens(self, features):
        "Extract the Register tokens from the features"
        if self.num_register:
            cls_shift = int(self.has_cls)
            return {"register": features[0, cls_shift : cls_shift + self.num_register]}
        return {}

    def extract_patch_tokens(self, features):
        "Extract the patch tokens from the features"

        cls_shift = int(self.has_cls)
        return {"patch": features[0, cls_shift + self.num_register :]}
