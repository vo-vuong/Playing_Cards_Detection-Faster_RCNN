import os
import re
from constants.paths_const import TEST_OUTPUT_PATH


# Get the last file and create a new file name
def get_new_file(output_type):
    folder_path = TEST_OUTPUT_PATH
    if output_type == "images":
        file_extension = ".jpg"

    file_list = os.listdir(folder_path)
    # get a list of files with numeric names
    numeric_file_list = [
        file for file in file_list if re.match(r"^\d+{}$".format(file_extension), file)
    ]

    if numeric_file_list:
        latest_file = max(
            numeric_file_list, key=lambda x: int(x[: -len(file_extension)])
        )
        new_file_name = (
            str(int(os.path.splitext(os.path.basename(latest_file))[0]) + 1)
            + file_extension
        )
        new_file_path = os.path.join(folder_path, new_file_name)

        return new_file_path
    else:
        new_file_path = os.path.join(folder_path, "1" + file_extension)
        return new_file_path
