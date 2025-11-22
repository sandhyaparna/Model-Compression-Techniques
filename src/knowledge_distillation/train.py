import torch
import utils
from fpn_feature_hooks import FeatureHook, FPNAdapter, RPNHeadHook
from kd_losses import (
    mse_feature_loss_from_lists,
    pair_feats_by_size,
    rpn_objectness_distill,
)



# ==============================================================================
# Training loop (KD) - Teacher → Hooks → Adapter → KD Loss → Student
# ==============================================================================
def train_one_epoch_kd(
    teacher,
    student,
    optimizer,
    lr_scheduler,
    data_loader,
    device,
    epoch, KD_T, LAMBDA_FPN, LAMBDA_RPN, FEATURE_NORM,
    adapter: FPNAdapter,
    teacher_backbone_hook: FeatureHook,
    student_backbone_hook: FeatureHook,
    teacher_rpn_hook: RPNHeadHook,
    student_rpn_hook: RPNHeadHook,
    print_freq=100,
):
    """
    Trains a student object detection model for one epoch using Knowledge Distillation (KD) from a teacher model.
    Combines supervised detection loss with KD losses from intermediate features and RPN objectness logits.
    **Steps**
        1.  **Setup**: Move models to device, set student to train mode.
        2.  **Loop over batches**:
            -   Move images and targets to device.
            -   Zero gradients.
        3.  **Teacher Forward Pass**:
            -   Run teacher model under `torch.no_grad()`.
            -   Capture backbone features and RPN logits via hooks.
        4.  **Student Forward Pass**:
            -   Compute supervised detection losses.
            -   Capture backbone features and RPN logits.
        5.  **Feature Pairing & Adaptation**:
            -   Pair student and teacher FPN levels by spatial size.
            -   Initialize adapter if needed.
            -   Align student channels to teacher channels.
        6.  **Compute KD Losses**:
            -   Feature KD: MSE between normalized student and teacher features.
            -   RPN KD: BCE loss on objectness logits with temperature scaling.
        7.  **Combine Losses**:
            -   Total loss = detection loss + weighted KD losses.
        8.  **Backpropagation & Update**:
            -   Backpropagate combined loss.
            -   Update weights and learning rate.
        9.  **Logging**:
            -   Track total loss, detection loss, KD losses, and LR.
    """
    teacher.to(device)
    student.to(device)
    student.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Clears previous gradients
        optimizer.zero_grad(set_to_none=True)

        # 1) Teacher forward (no grad) to capture features + RPN logits
        with torch.no_grad():
            teacher(images)  # triggers hooks

        teacher_feats_dict = teacher_backbone_hook.last
        teacher_rpn_obj = teacher_rpn_hook.objectness or []

        # 2) Student forward (with grad) to get supervised losses + capture hooks
        student_losses = student(images, targets)
        student_feats_dict = student_backbone_hook.last
        student_rpn_obj = student_rpn_hook.objectness or []

        # 3) Pair features by closest spatial size and adapt channels
        s_list, t_list = pair_feats_by_size(
            student_feats_dict, teacher_feats_dict, max_logdist=1.1
        )
        if not adapter._initialized:
            if len(s_list) == 0 or len(t_list) == 0:
                raise RuntimeError(
                    "No overlapping FPN levels could be paired. "
                    "Increase max_logdist in pair_feats_by_size if needed."
                )
            adapter.init_from_feature_shapes(s_list, t_list)
        s_aligned_list = adapter(s_list)

        # 4) KD losses
        # Feature KD: MSE between normalized student and teacher features.
        kd_feat = mse_feature_loss_from_lists(s_aligned_list, t_list, normalize=FEATURE_NORM)
        # RPN KD: Distills objectness logits using temperature-scaled BCE loss.
        kd_rpn  = rpn_objectness_distill(student_rpn_obj, teacher_rpn_obj, T=KD_T, max_logdist=1.1)

        # Total = supervised detection loss + weighted KD losses.
        total_det_loss = sum(loss for loss in student_losses.values())
        total_loss = total_det_loss + LAMBDA_FPN * kd_feat + LAMBDA_RPN * kd_rpn

        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Logging
        loss_dict = {k: v.detach().item() for k, v in student_losses.items()}
        metric_logger.update(
            loss=total_loss.item(),
            det_loss=total_det_loss.item(),
            kd_fpn=kd_feat.item(),
            kd_rpn=kd_rpn.item(),
            lr=optimizer.param_groups[0]["lr"],
            **loss_dict,
        )

    return {k: v.global_avg for k, v in metric_logger.meters.items()}
