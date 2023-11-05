# Camera Calibration

1. change settings in `calibration_settings.yaml`.

|Setting| Description|
|---|---|
camera0| This setting indicates the identifier or index for the first camera, often referred to as the left camera.
camera1| This setting indicates the identifier or index for the second camera, typically the right camera in a stereo setup.
frame_width| It specifies the width of the camera frames in pixels.
frame_height| This setting determines the height of the camera frames in pixels. 
mono_calibration_frames| This parameter specifies the number of frames captured for monocular (single camera) calibration.
stereo_calibration_frames| This parameter indicates the number of frames captured for stereo calibration.
view_resize| It determines whether the captured frames or images should be resized for calibration.
checkerboard_box_size_scale| This setting represents the scaling factor applied to the size of the squares on the checkerboard pattern. The size of each square on the checkerboard is adjusted by multiplying the default size by 3.5 times. This scaling factor helps in calibrating the camera accurately.
checkerboard_rows| It specifies the number of rows (horizontal) on the checkerboard pattern used for calibration.
checkerboard_columns| This parameter indicates the number of columns (vertical) on the checkerboard pattern used for calibration.
cooldown| This setting represents the cooldown time between capturing frames during calibration.

2. run the `calibrate.py` script:

```bash
python calibrate.py calibration_settings.yaml
```