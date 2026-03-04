#! /usr/bin/env python

import os

'''
write_to_file write a string to a file, only if the path to the file does not exist
'''
def write_to_file(path,string):
    if not os.path.exists(path):
        with open(path, "w") as file:
            file.write(string)

def write_settings(path,args):
    string = ""
    for e in args.items():
        string += "{} {}\n".format(e[0], e[1])
    write_to_file(path,string)