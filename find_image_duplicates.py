import pprint
from os.path import relpath
from pathlib import Path
from filehash import FileHash
import glob

basedir = Path("C:\\Labs\\DataNoDuplicates\\")  # "R:\\mp3\\שחר סגל ורועי בר נתן\\גלי צה_ל\\"


def insert_file_to_dict(hashdict: dict, hasher, file):
    file_hash = hasher.hash_file(file)
    if file_hash not in hashdict:
        hashdict[file_hash] = []
    shortfile = relpath(file, basedir)
    hashdict[file_hash].append(shortfile)


if __name__ == "__main__":
    files_by_hash = {}
    md5hasher = FileHash('md5')
    files = glob.glob(str(basedir) + '/**/*.png', recursive=True)

    for i, file in enumerate(files):
        # if i>100:
        #     break;
        insert_file_to_dict(files_by_hash, md5hasher, file)

    total_duplicate_types = 0
    total_duplicates = 0
    for hash, images in files_by_hash.items():
        if len(images) > 1:
            print(images)
            total_duplicate_types = total_duplicate_types + 1
            total_duplicates = total_duplicates + len(images)

    print()
    print(f'Base directory: {basedir}')
    print(f'Total different images: {len(files_by_hash.keys())},Total duplicate instances count: {total_duplicate_types}, different images count: {total_duplicates}')


