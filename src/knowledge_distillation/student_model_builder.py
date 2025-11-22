from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FCOS_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def build_student(num_classes: int, variant="frcnn_mobilenet"):
    """
    Build a student detection model with a specified head for `num_classes`.

    Conventions:
      - Faster R-CNN / SSD / SSDLite expect num_classes INCLUDING background (K+1).
      - RetinaNet / FCOS expect num_classes WITHOUT background (K).
    """

    if variant == "frcnn_mobilenet":
        # Faster R-CNN (MobileNetV3 + FPN, 800px short side)
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        )
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        return model

    elif variant == "frcnn_mobilenet_320":
        # Faster R-CNN (MobileNetV3 + FPN, 320px short side)
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        )
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        return model

    elif variant == "retinanet_r50":
        # RetinaNet: no background class in the head (K only)
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=model.head.classification_head.conv[0].in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes - 1,  # RetinaNet has no background class
        )
        return model

    elif variant == "fcos_r50":
        # FCOS: no background class in the head (K only)
        model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
        in_channels = model.head.cls_logits.in_channels
        model.head = FCOSHead(in_channels, num_classes=num_classes - 1)
        return model

    elif variant == "ssd300_vgg16":
        # SSD300 (VGG16 backbone)
        # Torchvision constructors support num_classes; if it differs from
        # pretrained categories, the classification head is reinitialized.
        model = ssd300_vgg16(
        # weights=SSD300_VGG16_Weights.COCO_V1,
            weights=None,  # <- don't load COCO detector head
            weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,  # pretrained backbone
            num_classes=num_classes,  # INCLUDE background (K+1)
        )
        return model

    elif variant == "ssdlite320_mobilenet":
        # SSDLite320 (MobileNetV3 Large backbone)
        model = ssdlite320_mobilenet_v3_large(
            weights=None,  # <- don't load COCO detector head
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,  # pretrained backbone
            num_classes=num_classes,  # INCLUDE background (K+1)
        )
        return model

    else:
        raise ValueError(
            "Unknown student variant. "
            "Valid: {'frcnn_mobilenet', 'frcnn_mobilenet_320', 'retinanet_r50', "
            "'fcos_r50', 'ssd300_vgg16', 'ssdlite320_mobilenet'}"
        )
