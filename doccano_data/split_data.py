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

def split_data(data):
    for line in data:
        if line["annotations"]:
            dump_jsonl([line], './annotated.jsonl', append=True)
        else:
            dump_jsonl([line], './unannotated.jsonl', append=True)

full_data = load_jsonl('./project_1_dataset.jsonl')
split_data(full_data)
print('Complete.')