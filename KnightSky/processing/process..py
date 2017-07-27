import os
import re


original_data = "ficsgamesdb.pgn"
original_path = os.path.join(os.pardir, os.pardir, original_data)
print(original_path)

processed_data = "processed.pgn"
processed_path = os.path.join(os.pardir, os.pardir, processed_data)

print(processed_path)


def remove_metadata():

    forfeit = "forfeits by disconnection"
    with open(original_path, 'r') as raw, open(processed_path, 'w+') as processed:
        for line in raw:
            if line[:2] == "1." and forfeit not in line:
                end = line.index('{')
                processed_line = line[:end].replace('+', '').replace('#', '') + '\n'
                processed_line = re.sub(r'\d\.\s', '', processed_line)
                processed.write(processed_line)


def convert_to_bitmap():
    with open(processed_path, 'r') as processed:
        for line in processed:
            move_list = line.split(' ')

if __name__ == '__main__':
    remove_metadata()