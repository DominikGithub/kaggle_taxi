#!/usr/bin/env /home/dominik/anaconda2/bin/python

import statics as st
from utils_file import load_csv

# class HashSet():
#     set = []
#     def add(self, value):
#         set.append(value)
#
#     def contains(self, value):
#         idx = set.index()
#         if idx >= 0:
#             return idx
#         else:
#             return -1
#
#     def indexOf(self, value):
#         return set.index(value)

class HashMap(dict):
    def __setitem__(self, key, value):
        if key not in self:
            dict.__setitem__(self, key, value)
        else:
            raise Exception("Key already exists")   #KeyError

    def getkey_or_create(self, item):
        '''
        get key for given item, and create new entry if not found
        '''
        try:
           key = dict.__getitem__(self, item)
        except:
            key = len(dict.keys(self))
            dict.__setitem__(self, key, item)
        finally:
            return key


class IncMap(HashMap):
    def incrementAt(self, key):
        if key not in self:
            dict.__setitem__(self, key, 1)
        else:
            dict.__setitem__(self, key, dict.__getitem__(self, key)+1)


def create_distr_Map():
    dist_map = HashMap()
    for district in load_csv(st.data_dir + 'cluster_map'):
        dist_map.__setitem__(district[0], district[1])
    return dist_map