import csv
import os
from datetime import datetime


input_text_file = 'prices.txt'
output_csv_file = 'prices.csv'
temp_file = "temp.txt"

with open(input_text_file, "r") as infile, open(temp_file, "w") as outfile:
    for line in infile:
        if "Dividend" and "Stock" not in line:
            outfile.write(line)
os.replace(temp_file, input_text_file)

data_to_write = []
with open(input_text_file, 'r') as infile:
    for line in infile:
        values = line.strip().split('\t')
        date_parts = values[0].strip().split(', ')
        month_number = datetime.strptime(date_parts[0][0:3], '%b').month
        day_number = date_parts[0][4:]
        year_number = date_parts[1]
        values[0] = str(month_number) + "/" + str(day_number) + "/" + str(year_number)
        values[-1] = values[-1].replace(',','')
        temp = values[5]
        values[5] = values[6]
        values[6] = temp

        data_to_write[:0] = [values]

with open(output_csv_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(data_to_write)

print(f"Data from '{input_text_file}' has been written to '{output_csv_file}' successfully.")