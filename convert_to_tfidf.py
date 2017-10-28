#load in json files
#each file represents a topic, or section.
directory = "articles"
i = 0
articles = []
topics_dict = {}
for filename in os.listdir(directory):
    topics_dict[i] = filename.split('.')[0]
    if filename.endswith(".json"):
        with open(join(directory, filename), 'rb') as f:
            json_data = f.read()
            data = json.loads(json_data)
            for key, value in data.items():
                articles.append([value, i])
        i += 1
