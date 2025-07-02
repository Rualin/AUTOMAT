# AUTOMAT
Repository for helping poor people

### yolo_add_metric.txt:
There are writen functions to change if you want to add new metric in ultralitics

### my_yolo.yaml:
My yolo config with two cut-out scales


## Thoughts:
- Add SPPF/SPP in the end of backbone for better tracking of general information
- Switch C3K2 to C2F (only because in company use yolov8, not yolo11)
- Switch Convs to DWS Convs (both pure convs and convs inside the C3K2 block)
- It may be beneficial to return the medium scale
