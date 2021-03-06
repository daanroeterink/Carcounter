# Carcounter
Simple hobby project to count cars

Coordinates are from 1920x1080 camlink connected to fullhd camera output at 60fps

# References
Uses sort.py from https://github.com/abewley/sort this needs to be available before the carcounter can run

# Requirements

 * sort (mentioned above)
 * yolov4.cfg and yolov4.weigths from https://github.com/AlexeyAB/darknet
 * opencv installed (use cuda support https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)

```bash
├── Carcounter
│   ├── carCounter.py
│   ├── LICENSE
│   ├── README.md
│   ├── yolov4.cfg
│   └── yolov4.weights
└── sort
    ├── data
    ├── LICENSE
    ├── __pycache__
    ├── README.md
    ├── requirements.txt
    └── sort.py
```





