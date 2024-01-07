import time
import csv
import os


def result_append(row, config):
    # rank = config['rank']
    file_path = '../result_stat/'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_path = file_path + config['dataset'] + '.csv'
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # if not file_exists:
        #     csv_writer.writerow(['ACC', 'AUC', 'F1', 'DP', 'EO', 'rank', 'time', 'config'])
        csv_writer.writerow(row)
        csv_file.flush()