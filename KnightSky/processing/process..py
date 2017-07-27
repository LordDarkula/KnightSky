import os


original_data = "ficsgamesdb.pgn"
original_path = os.path.join(os.pardir, os.pardir, original_data)
print(original_path)

processed_data = "processed.pgn"
processed_path = os.path.join(os.pardir, os.pardir, processed_data)

print(processed_path)

forfeit = "forfeits by disconnection"
with open(original_path, 'r') as raw, open(processed_path, 'w+') as processed:
    for line in raw:
        if line[:2] == "1." and forfeit not in line:
            end = line.index('{')
            processed.write(line[:end] + '\n')
