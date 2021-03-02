import json

def load_jsonl(input_path, verbose=False):
    'Read from JSON lines file'
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    if verbose:
        print(f'Loaded {len(data)} lines from {input_path}')
    return data

def dump_jsonl(data, output_path, append=False, verbose=False):
    'Write data to JSON lines file'
    mode = '+a' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line + '\n')
    if verbose:
        print(f'Wrote {len(data)} lines to {output_path}')

def split_data(data, verbose=False):
    num_lines = [0, 0]
    for line in data:
        if line["annotations"]:
            num_lines[1] += 1
            dump_jsonl([line], './Mar_1_annotated.jsonl', append=True)
        else:
            num_lines[0] += 1
            dump_jsonl([line], './Mar_1_unannotated.jsonl', append=True)
    if verbose:
        print(f'Split data, {num_lines[1]} annotated, {num_lines[0]} not annotated.')

full_data = load_jsonl('./Mar_1_Dataset.jsonl')
split_data(full_data, verbose=True)
print('Complete.')