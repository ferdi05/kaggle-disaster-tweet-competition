# kaggle-disaster-tweet-competition
Participating to a Kaggle competition without coding any Machine Learning

---
## Installation
You will need to install the 🤗 Inference API wrapper at [https://github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)

Run disaster_tweets.py with your own API token. You can get yours from [Hugging Face website](https://hf.co)

---

Associated to the blog post at 

Compared to the blog post, you will notice that I have added a clean version of the datasets. I tried to obtain better results by cleaning the train, validation and test datasets (removing URL, abbrevations....), but did not achieve a better ranking 🤷
See the cleaning script at clean_tweets.py