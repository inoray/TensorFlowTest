import csv
import sys

csv_file = open('./train.csv', 'rb')
f = csv.reader(csv_file)


noRow = 0
for row in f:
    print(row)
    noRow += 1

print(noRow)

csv_file.close()
