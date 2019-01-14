import os
import datetime
import time

def generate_csv(results_dir):
    results_files_list = os.listdir(results_dir)
    now = time.time()
    threshold = datetime.timedelta(hours=8) # can also be minutes, seconds, etc.

    results_list = []
    for file in results_files_list:
        file_path = results_dir+file
        filetime = os.path.getmtime(file_path) # filename is the path to the local file you are refreshing
        delta = datetime.timedelta(seconds=now - filetime)
        if delta < threshold:
            results_list.append(file_path)
            print(delta)
            print(file_path)

if __name__ == "__main__":
    results_dir = '/mnt/nariz/cazevedo/'
    generate_csv(results_dir)