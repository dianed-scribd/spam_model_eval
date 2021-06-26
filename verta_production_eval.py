import requests
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

headers = {
  'Access-token': '637c5339-0b90-48e9-9fe3-043ae75edf58',
  'Content-type': 'application/json',
}

"""
  Single sample prediction
"""

data = {"text": "",
        "title": "Vagabond #14",
        "description": "Vagabond Mangá #14 (Português)",
        "incentivized_upload": 0,
        "delta_days": 10,
        "is_facebook_user": 0,
        "page_count": 181,
        "extension": "pdf",
        "producer": "this issome producer",
        "checksums": ["lakjsdlfkjaslfdjkasldfj","lakjsdflakjsdflkjasdflkj"],
        "hyperlinks": ["www.scribd.com", "www.wikipedia.com"]
       }

response = requests.post('https://scribd.external.verta.ai/api/v1/predict/fragile-amaranth-sheep', headers=headers, json=data)

"""
  Batch sample prediction
"""

f = open ('/dbfs/FileStore/shared_uploads/dianed@scribd.com/spam_production_data_version_7.json', "r")

data = json.loads(f.read())

pred_labels = []
pred_probs = []

for i in range(len(data)):
    raw_input = data[i]
    if not raw_input['description']:
        raw_input['description'] = ''
    
    response = requests.post('https://scribd.external.verta.ai/api/v1/predict/fragile-amaranth-sheep', headers=headers, json=raw_input)

    pred_probs.append(json.loads(response.text)['probability'])
    pred_labels.append(1 if json.loads(response.text)['label']=='spam' else 0)


true_labels = []
for i in range(len(data)):
    if data[i]['spam_status'] == 'not_spam':
        true_labels.append(0)
    else:
        true_labels.append(1)


print(precision_recall_fscore_support(true_labels, pred_labels, average='binary'))
print(roc_auc_score(true_labels, pred_probs))
print(accuracy_score(true_labels, pred_labels))
