API_TOKEN = "api_BeypKrKRhrvwITpiaUzFVYuuadOcoqYeIO"


import requests, csv

API_URL = "https://api-inference.huggingface.co/models/ferdinand/autonlp-kaggle-competition-clean-6421399" # URL of our model
MODEL_MAX_LENGTH = 512 # parameter of our model, can be seen in config.json at "max_position_embeddings"
headers = {"Authorization": f"Bearer {API_TOKEN}"} # our API token

fr = open("~/assets/test_clean.csv") # our test data
csv_read = csv.reader(fr)
next(csv_read) # skipping the header row

fw = open("~/assets/submission.csv", "w", encoding="UTF8") # our predictions data
csv_write = csv.writer(fw)
csv_write.writerow(['id', 'target']) # writing the header row

#returns a label : about a disaster or not given a tweet content
def run(tweet_content):

    # API payload
    payload = {
        "inputs": tweet_content[:MODEL_MAX_LENGTH], # send the tweet content to the API, possibly truncated to meet our model requirements
        "options": {"wait_for_model": True} # sometimes the API may take a little while to respond, we will wait for it to be ready
    }

    # calling the API
    response = requests.post(API_URL, headers=headers, json=payload)
    answer = response.json()

    # Determining which label to return according to the prediction with the highest score
    # example of an API call response: [[{'label': '0', 'score': 0.9159180521965027}, {'label': '1', 'score': 0.08408192545175552}]]
    max_score = 0
    max_label = None 
    for dic in answer[0]:
        for label in dic['label']:
            score = dic['score']
            if score > max_score:
                max_score = score
                max_label = label
    return max_label


for row in csv_read: # call the API for each row

    # writing in the submission file the tweet ID and its associated label: about a disaster or not
    write_row = [row[0], run(row[1])] # row[0] is the tweet ID, row[1] is the tweet content
    csv_write.writerow(write_row)
