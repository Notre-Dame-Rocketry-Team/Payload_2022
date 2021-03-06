# NDRT 2021-2022
# Hector Juarez ACS subteam
# This code contains functions to create CSV files and add data to them
#
# Note that each of the following functions opens/closes a csv file every time it is called,
# as well as creates a new csv writer object. Perhaps consider moving these steps outside of the functions
# upon code integration, so that these steps are only done once when the code is run. 

# Also, these two functions are identical, but named differently for ease of use and clarity

import csv

def flatten(sequence):
    if len(sequence) == 0:
        return sequence
    if isinstance(sequence[0], list):
        return flatten(sequence[0]) + flatten(sequence[1:])
    return sequence[:1] + flatten(sequence[1:])

# creates a new CSV file
# input: filepath to new CSV file, list of headers as a python list
# returns: n/a
def newCSV(filepath,headers):
    with open(filepath, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(headers)

# adds a row of data to a given CSV file
# input: filepath to new CSV file, row of data as a python list
# output: n/a
def addRow(filepath,row):
    # open file in write mode
    with open(filepath, 'a', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(row)
