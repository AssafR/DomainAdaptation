import os
import pprint
from os.path import relpath
from pathlib import Path
from filehash import FileHash
import glob
import itertools

basedirs = ["C:\\Labs\\DataWithGender\\"]  # "C:\\Labs\\DataNoDuplicates\\"
priority_dir = 'DataWithGender'


def insert_file_to_dict(basedir: str, hashdict: dict, hasher, file):
    file_hash = hasher.hash_file(file)
    if file_hash not in hashdict:
        hashdict[file_hash] = []
    shortfile = relpath(file, basedir)
    hashdict[file_hash].append(file)


def choose_one_file_by_priority(list):
    if len(list) == 0:
        return None, []
    chosen = list[0]
    for item in list:
        if priority_dir in item:
            chosen = item
            break
    return chosen, [it for it in list if it!=chosen]


if __name__ == "__main__":
    files_by_hash = {}
    md5hasher = FileHash('md5')

    for basedir in basedirs:
        files = glob.glob(basedir + '/**/*.png', recursive=True)

        for i, file in enumerate(files):
            # if i>100:
            #     break;
            insert_file_to_dict(basedir, files_by_hash, md5hasher, file)

    total_duplicate_types = 0
    total_duplicates = 0
    for hash, images in files_by_hash.items():
        if len(images) > 1:
            #print(images)
            total_duplicate_types = total_duplicate_types + 1
            total_duplicates = total_duplicates + len(images)
            # chosen, unchosen = choose_one_file_by_priority(images)
            for to_delete in images:
                os.remove(to_delete)

    print()
    print(f'Base directories: {basedirs}')
    print(
        f'Total different images: {len(files_by_hash.keys())},Total duplicate instances count: {total_duplicate_types},'
        f'different images count: {total_duplicates}')
