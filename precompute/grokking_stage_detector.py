"""
Automatic detection of grokking stage boundaries from training curves.

Three stages:
  Stage 1 (Memorization): Train accuracy rises, test accuracy stays low
  Stage 2 (Transition): Test accuracy starts climbing
  Stage 3 (Generalization): Test accuracy near 1.0

Returns (stage1_end, stage2_end) as epoch indices, or (None, None) if
grokking is not detected.
"""


def detect_grokking_stages(train_losses, test_losses, train_accs=None, test_accs=None):
    """
    Detect memorization -> transition -> generalization boundaries.

    Heuristic:
    - stage1_end: first epoch where train accuracy >= 0.95 (memorization complete)
    - stage2_end: first epoch where test accuracy >= 0.95 (generalization reached)

    Fallback (if accuracy curves not available):
    - stage1_end: first epoch where train loss < 0.1
    - stage2_end: first epoch where test loss < 0.1
    """
    if train_accs is not None and test_accs is not None:
        stage1_end = None
        for i, a in enumerate(train_accs):
            if a >= 0.95:
                stage1_end = i
                break

        stage2_end = None
        for i, a in enumerate(test_accs):
            if a >= 0.95:
                stage2_end = i
                break

        # Sanity: stage1 should come before stage2
        if stage1_end is not None and stage2_end is not None and stage1_end >= stage2_end:
            stage1_end = stage2_end // 3
    else:
        stage1_end = None
        for i, loss in enumerate(train_losses):
            if loss < 0.1:
                stage1_end = i
                break

        stage2_end = None
        for i, loss in enumerate(test_losses):
            if loss < 0.1:
                stage2_end = i
                break

    return stage1_end, stage2_end
