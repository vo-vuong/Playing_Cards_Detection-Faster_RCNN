import requests
import os
from tqdm import tqdm
from constants.paths_const import TRAINED_MODEL_PATH


def download_best_model():
    # Download the model from the release on github
    file_url = "https://github.com/vo-vuong/playing_cards_detection-faster_rcnn/releases/download/v1.0.0/best.pt"
    save_path = os.path.join(TRAINED_MODEL_PATH, "best.pt")

    response = requests.get(file_url, stream=True)

    # Check if the download request was successful or not
    if response.status_code == 200:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 1024

        print("Start downloading: best.pt (v1.0.0) at: ", file_url)
        # Show progress bar
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, colour="green")

        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))

        progress_bar.close()
        print("Download successfully, the file is saved at: ", save_path)
    else:
        print("Download failed.")
