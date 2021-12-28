import json
import os

def load_option(opt_path):
    with open(opt_path, 'r') as json_file:
        json_obj = json.load(json_file)
        return json_obj

