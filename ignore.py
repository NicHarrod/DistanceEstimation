from sam3.model_builder import build_sam3_image_model 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_sam3_image_model(device=str(device), enable_inst_interactivity=True, load_from_HF=True) 
print(model.backbone)

# output:
# (apes) PS C:\Users\nicpa\OneDrive\Desktop\Uni-Stuff\Year 4\Chimps\distance-estimation> python .\sam_test.py ..\small_example\adk01\transects\transect1\detection_frames\8230044_0.jpg
# C:\Users\nicpa\OneDrive\Desktop\Uni-Stuff\Year 4\Chimps\sam3\sam3\model_builder.py:8: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
#   import pkg_resources
# Running SAM3 on image of shape (720, 1280, 3) with 1 boxes
# Traceback (most recent call last):
#   File "C:\Users\nicpa\OneDrive\Desktop\Uni-Stuff\Year 4\Chimps\distance-estimation\sam_test.py", line 163, in <module>
#     run_and_visualize(
#   File "C:\Users\nicpa\OneDrive\Desktop\Uni-Stuff\Year 4\Chimps\distance-estimation\sam_test.py", line 89, in run_and_visualize
#     sam3_masks = sam3_model(sam3_img, sam3_boxes)
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\nicpa\OneDrive\Desktop\Uni-Stuff\Year 4\Chimps\distance-estimation\sam3_wrapper.py", line 66, in __call__
#     self.predictor.set_image(img_rgb)
#   File "C:\Users\nicpa\miniconda3\envs\apes\Lib\site-packages\torch\utils\_contextlib.py", line 116, in decorate_context
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\nicpa\OneDrive\Desktop\Uni-Stuff\Year 4\Chimps\sam3\sam3\model\sam1_task_predictor.py", line 101, in set_image
#     backbone_out = self.model.forward_image(input_image)
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\nicpa\OneDrive\Desktop\Uni-Stuff\Year 4\Chimps\sam3\sam3\model\sam3_tracker_base.py", line 446, in forward_image
#     backbone_out = self.backbone.forward_image(img_batch)["sam2_backbone_out"]
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AttributeError: 'NoneType' object has no attribute 'forward_image'