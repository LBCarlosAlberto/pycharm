import json
from os.path import join, exists

def read_single_file(path, filename, target_file):
    filepath = join(path, filename)
    written_file = open(target_file, 'a')
    with open(filepath, 'rb') as f:
        json_data = f.read()
        articles = json.loads(json_data)
        for art in articles:
            dic = {art['id']:art['fields']['bodyText']}
            json.dump(dic,written_file)
    written_file.close()
    
if __name__ == '__main__':
    directory = "D:/tempdata/articles"
    target_file = #output filename here name is the section name, like 'politics'
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            read_single_file(directory, filename, target_file)
