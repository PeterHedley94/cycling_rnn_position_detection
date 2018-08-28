import json
import os

def reformat_json(my_file):
    my_data = {}
    my_data['data'] = []
    with open(my_file) as f:
        for line in f:
            my_data['data'].append(json.loads(line))
    return my_data

def extract_filename_data(filename):
    my_split = filename.split('_')
    return my_split




# my_data = {}
# my_data['data'] = []
# with open('test.json') as f:
#     for line in f:
#         my_data['data'].append(json.loads(line))
# print(my_data)
#
# with open('final_test.json', 'w') as outfile:
#     json.dump(my_data, outfile, indent=4)