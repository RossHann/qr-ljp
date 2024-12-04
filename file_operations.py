import json


def get_json_content(file_path):
    with open(file_path, 'r') as f:
        content = json.load(f)
    f.close()
    return content


def set_json_file(content, file_path):
    with open(file_path, 'w') as f:
        json.dump(content, f, ensure_ascii=False)
    f.close()

def get_jsonl_content(file_path):
    content = []
    with open(file_path, "r") as file:
        for line in file:
            obj = json.loads(line)
            content.append(obj)
    return content


def set_jsonl_file(content, file_path):
    with open(file_path, "w") as file:
        for obj in content:
            line = json.dumps(obj, ensure_ascii=False)
            file.write(line + "\n")
