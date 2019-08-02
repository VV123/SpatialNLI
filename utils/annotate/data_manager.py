__author__ = 'Jingjing Li'

import os
import re
import pandas as pd 
from StringIO import StringIO
path = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(path).replace('utils/annotate',
                                          'data/base/geo/')
GEO_TABLE = os.path.join(FILE_PATH, 'geobase.sql')
GEO_KEY1 = os.path.join(FILE_PATH, 'key_w.txt')
GEO_KEY2 = os.path.join(FILE_PATH, 'human_info_w.txt')
GEO_KEY3 = os.path.join(FILE_PATH, 'special_w.txt')

FILE_PATH_REST = os.path.dirname(path).replace('utils/annotate',
                                          'data/base/rest/')
FILE = os.path.join(FILE_PATH_REST, 'rest-db-name')
FILEKEY = os.path.join(FILE_PATH_REST, 'words.txt')
FILEHUMAN = os.path.join(FILE_PATH_REST, 'human_info_w.txt')

def _read_dump(dump_filename, target_table):
    sio = StringIO()

    fast_forward = True
    with open(dump_filename, 'rb') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('insert') and target_table in line:
                fast_forward = False
            if fast_forward:
                continue
            data = re.findall('\([^\)]*\)', line)
            try:
                newline = data[0]
                newline = newline.strip(' ()')
                newline = newline.replace('`', '')
                sio.write(newline)
                sio.write("\n")
            except IndexError:
                pass
            if line.endswith(';'):
                break
    sio.pos = 0
    return sio

def _import_data(db_name, col_name, file = GEO_TABLE):
    table_data = _read_dump(file, db_name)
    table = pd.read_csv(table_data)
    names = list(table[col_name])
    name_ = []
    for name in names:
        name = name.replace(" '",'')
        name = name.replace("'",'')
        name_.append(name)
    name = name_

    return name

def _import_data_rest(target_table,filename = FILE):
    res = []
    with open(filename, 'rb') as f:
        for line in f:
            line = line.replace('food_type','foodtype')
            if line[0:len(target_table)] == target_table:
                line = line.replace('foodtype','food_type')
                line = line.strip()
                data = re.match(r'^.*\(\[.*\].*\[.*\], \'(.*)\'\).*$', line)
                try:
                    newline = data.group(1)
                    newline = newline.replace('-',' ')
                    newline = newline.replace('_',' ')
                    newline = newline.replace('?','')
                    newline = newline.replace("'",'')
                    newline = newline.replace("[",' [ ')
                    newline = newline.replace("]",' ] ')
                    newline = newline.replace('( * )','')
                    res.append(newline)
                except IndexError:
                    pass
    return res

def _read_list_special(filename,flag,reverse):
    dic = {}
    f = open(filename, 'r')
    s = f.read().split('\n')
    for line in s:
        line = line.split(':')
        word = line[0]
        if flag == 'special':
            line_ = line[1].split(',')
            dic[word] = line_
        elif flag == 'human_info':
            line_ = line[1].split(',')
            dic[word] = line_
    f.close()
    return dic

def _read_list(filename,dic, reverse):
    f = open(filename, 'r')
    s = f.read().split('\n')
    s_ = []
    for line in s:
        line = line.replace("'",'')
        s_.append(line)
    s_.sort(key=len, reverse = reverse)
    f.close()
    dic += s_
    return dic

def read_word_geo():

    city = _import_data('city', ' name')
    city += _import_data('state', ' capital')
    state_name = _import_data('state', 'name')
    state_abbre = _import_data('state', ' abbreviation')
    state_place = _import_data('state', ' highest_point')
    state_place += _import_data('state', ' lowest_point')
    country = _import_data('country', 'name')
    lake = _import_data('lake', 'name')
    mountain = _import_data('mountain', 'name')
    river = _import_data('river', 'name')

#some words such as the 'arkansas' river, appears in the state place table as 'arkansas river', remove this kind of word out of state_place
    state_place = list(set(state_place))
    for i_state in state_place:
        for i_river in river:
            if i_river + ' river' == i_state:
                state_place.remove(i_state)

        for i_lake in lake:
            if 'lake ' + i_lake == i_state:
                state_place.remove(i_state)

    const_w = city + state_name + state_place + lake + country + mountain + river
    const_w = list(set(const_w))
    const_w += ['the united states','united states','america','us','the us','the usa','country','the country']

    #load key word
    key_w = []
    key_w = _read_list(GEO_KEY1,key_w, True)

    human_info = {}
    human_info = _read_list_special(GEO_KEY2,'human_info',True)

    special_w = {}
    special_w = _read_list_special(GEO_KEY3,'special',True)

    return const_w, key_w, human_info, special_w, city, state_name, state_abbre, state_place, country, lake, mountain, river


def read_word_rest():
    #load key word
    street = _import_data_rest('street')
    city = _import_data_rest('city')
    county = _import_data_rest('county')
    region = _import_data_rest('region')
    rest = _import_data_rest('rest')
    foodtype = _import_data_rest('foodtype')
    rating = _import_data_rest('rating')

    key_w = []
    key_w = _read_list(FILEKEY,key_w, True)

    human_info = {}
    human_info = _read_list_special(FILEHUMAN,'human_info',True)

    return key_w, human_info, street, city, county, region, rest, foodtype, rating




