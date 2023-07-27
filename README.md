# Playing Cards Detection using Faster-RCNN

## Introduction

The project is an application to detect and classify a deck of 52 playing cards using the Faster R-CNN model.

For each suffix followed by a label: H: `Hearts`, D: `Diamonds`, C: `Clubs`, S: `Spades`.

<p align="center">
  <img src="https://raw.githubusercontent.com/vo-vuong/assets/main/playing_cards_detection-faster_rcnn/v1.0.0/outputs/1.jpg" width=300 height=300>
  <img src="https://raw.githubusercontent.com/vo-vuong/assets/main/playing_cards_detection-faster_rcnn/v1.0.0/outputs/2.jpg" width=300 height=300>
  <br/>
  <i>demo</i>
</p>

<details open>
<summary>Install</summary>

To install and run the project, follow these steps.

1. Clone the project from the repository:

```bash
git clone https://github.com/vo-vuong/playing_cards_detection-faster_rcnn.git
```

2. Navigate to the project directory:

```bash
cd playing_cards_detection-faster_rcnn
```

3. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # For Linux/Mac
.venv\Scripts\activate  # For Windows
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

</details>

<details open>
<summary>Inference</summary>

Run the inference by an image file and the result will be saved at `outputs/images/`.

```bash
python test.py --test_images img.jpg          # image
```

</details>

## Project Structure

```
playing_cards_detection-faster_rcnn/
├── constants
│   ├── config_const.py
│   └── paths_const.py
├── data
├── outputs                             # default path of model prediction
│   └── images
├── test_data
├── trained_models                      # the folder containing pretrain model
├── utils
│   ├── download_model.py
│   └── file_helpers.py
├── dataset_analysis.ipynb              # dataset analysis file
├── dataset.py                          # setup dataset for training
├── README.md
├── requirements.txt
├── test.py                             # main file to run test
└── train.py                            # training file
```

## Additional Resources

- [fasterrcnn_mobilenet_v3_large_fpn](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn)
- [Playing Cards Dataset](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)
