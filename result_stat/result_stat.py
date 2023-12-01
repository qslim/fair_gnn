import csv

input_file_path = 'pokec_z.csv'
# output_file_path = 'output_transposed.csv'

# Read all rows from the CSV file
with open(input_file_path, 'r', newline='') as input_csv:
    csv_reader = csv.reader(input_csv)

    csv_reader = sorted(csv_reader, key=lambda row: row[5])

    # Transpose the CSV data
    transposed_data = list(zip(*csv_reader))

# # Write the transposed data to a new CSV file
# with open(output_file_path, 'w', newline='') as output_csv:
#     csv_writer = csv.writer(output_csv)
#     csv_writer.writerows(transposed_data)
#
# print(f"CSV file transposed successfully. Transposed data saved to {output_file_path}")

print('&' + ' &'.join(transposed_data[0]) + ' \\'+'\\')
print('&' + ' &'.join(transposed_data[1]) + ' \\'+'\\')
print('&' + ' &'.join(transposed_data[2]) + ' \\'+'\\')
print('&' + ' &'.join(transposed_data[3]) + ' \\'+'\\')
print('&' + ' &'.join(transposed_data[4]) + ' \\'+'\\')
# print('&' + ' &'.join(transposed_data[4]))
