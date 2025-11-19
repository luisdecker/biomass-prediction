"Get pretrained models from the web"

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import timm


def get_hf_kit(model_name: str):
    "Get a pre-trained model, image_processor and tokenizer by name"

    if model_name.lower() == "convnext-dinov3-small":
        remote_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
        model = AutoModel.from_pretrained(remote_name)
        image_processor = AutoImageProcessor.from_pretrained(remote_name)
        return image_processor, None, model  # DINO model has no tokenizer

    if model_name.lower() == "vit-dinov3-small":
        remote_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        model = AutoModel.from_pretrained(remote_name)
        image_processor = AutoImageProcessor.from_pretrained(remote_name)
        # tokenizer = AutoTokenizer.from_pretrained(remote_name)
        return image_processor, None, model


def get_hf_image_processor(model_name: str):
    "Get a pre-trained image_processor by name"

    if model_name.lower() == "convnext-dinov3-small":
        remote_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
        image_processor = AutoImageProcessor.from_pretrained(remote_name)
        return image_processor


def get_timm_model(model_name: str):
    "Gets a pre-trained model using timm"

    if model_name.lower() == "convnext-dinov3-small":
        remote_name = "hf_hub:timm/convnext_small.dinov3_lvd1689m"
        model = timm.create_model(remote_name, pretrained=True)
        return model


def get_cnn_features(model_name: str):
    "Get the number of features in the output of a CNN feature extractor"

    features_by_model = {"convnext-dinov3-small": 768}
    return features_by_model[model_name.lower()]
