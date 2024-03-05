import pandas as pd
import time
from datetime import datetime

def save_data_to_csv(data, start_time):
    # Generate a filename based on the start and end time of the data
    end_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"data_{start_time}_{end_time}.csv"

    # Save the data to CSV
    data.to_csv(filename, index=False)

if __name__ == "__main__":
    # This script continuously monitors a file to determine when to save data
    save_file_path = "save_trigger.txt"

    # Check for the existence of the save trigger file
    while True:
        try:
            with open(save_file_path, "r") as trigger_file:
                trigger = trigger_file.read().strip()

            # If the trigger is "save", save the data and remove the trigger file
            if trigger == "save":
                data_to_save = pd.read_csv("shared_data.csv")
                save_data_to_csv(data_to_save, datetime.now().strftime("%Y%m%d%H%M%S"))
                with open(save_file_path, "w") as trigger_file:
                    trigger_file.write("saved")
            elif trigger == "stop":
                break
        except FileNotFoundError:
            # If the trigger file is not found, continue checking
            pass

        time.sleep(1)
