import json
from os.path import join, exists

def read_single_file(path, filename, dictionary):
    filepath = join(path, filename)
    with open(filepath, 'rb') as f:
        json_data = f.read()
        articles = json.loads(json_data)
        for art in articles:
            dictionary[art['id']] = art['fields']['bodyText']
    
if __name__ == '__main__':
    directory = "articles"
    target_file = 'politics.json'#output filename here name is the section name, like 'politics'
    dictionary = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            read_single_file(directory, filename, dictionary)
    with open(target_file, 'w') as f:
        json.dump(dictionary, f)
