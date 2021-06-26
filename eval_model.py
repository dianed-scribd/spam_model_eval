import json

import numpy as np
import re
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from urlextract import URLExtract
from urllib.parse import urlsplit

import os
import mlflow.sklearn
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.file_utils import TempDir
import joblib
import xgboost



f = open ('spam_production_data_version_8.json', "r")

data = json.loads(f.read())


class FeatureProcessor(object):

    MAX_TEXT_LENGTH = 10000
    ALPHA_NUMERIC_REGEX = r'\W+'

    def __init__(self):

        self.encoders = {}
        self.url_extractor = URLExtract()

        return

    def set_encoders(self, encoders):
        self.encoders = encoders
        return

    def skip(self, s):
        return s

    # Setup the encoders which consist of tfidf vectorizers and a one hot encoder
    def init_encoders(self):
        self.encoders = {
            'text': TfidfVectorizer(analyzer='word', tokenizer=self.skip, preprocessor=self.skip, token_pattern=None),
            'title': TfidfVectorizer(analyzer='word', tokenizer=self.skip, preprocessor=self.skip, token_pattern=None),
            'description': TfidfVectorizer(analyzer='word', tokenizer=self.skip, preprocessor=self.skip, token_pattern=None),
            'extension': OneHotEncoder(handle_unknown='ignore'),
            'producer': OneHotEncoder(handle_unknown='ignore'),
            'host': TfidfVectorizer(analyzer='word', tokenizer=self.skip, preprocessor=self.skip, token_pattern=None),
            'checksum': TfidfVectorizer(analyzer='word', tokenizer=self.skip, preprocessor=self.skip, token_pattern=None)
        }
        return

    # Construct the feature vector given all the input
    def feature_vector(self, text, title, description, incentivized_upload, delta_days, is_facebook, page_count, extension, producer, checksum, hyper_host):

        # Prepare features
        text = text[:self.MAX_TEXT_LENGTH]
        urls = self.url_list_from_text(text)
        host = self.url_host_names(urls)
        combined_hosts = list(set([*host, *hyper_host])) #Combine embedded and extracted URL hosts into one list
        
        #Extract producer from raw json
        pattern = "(?<=Producer:)(.*)(?=CreationDate:)"
        find_pattern = re.findall(pattern, producer)
        if not find_pattern:
            producer = ""
        else:
            producer = find_pattern[0].strip()
        
        #Just clean, DON'T TOKENIZE
        text_tokens = self.__clean_string(text)
        title_tokens = self.__clean_string(title)
        description_tokens = self.__clean_string(description)
        
        #Turn array columns into strings since encoders require strings, not tokens
        combined_hosts = ' '.join(combined_hosts)
        checksum = ' '.join(checksum)

        # Features
        f_text = self.transform([text_tokens], 'text')
        f_title = self.transform([title_tokens], 'title')
        f_description = self.transform([description_tokens], 'description')
        f_extension = self.transform([[extension]], 'extension')
        f_producer = self.transform([[producer]], 'producer')
        f_host = self.transform([combined_hosts], 'host')
        f_checksum = self.transform([checksum], 'checksum')

        return sparse.hstack([
            f_text,
            f_title,
            f_description,
            f_extension,
            self.to_matrix(page_count),
            self.to_matrix(incentivized_upload),
            self.to_matrix(len(urls)),
            self.to_matrix(len(hyper_host)),
            self.to_matrix(is_facebook),
            f_producer,
            f_host,
            f_checksum,
            self.to_matrix(np.log(max(0,delta_days) + 1))
        ])


    def text_clean_tokenize(self, text):
        all_text = self.__clean_string(text)
        return self.__token_splitter(all_text)

    def url_list_from_text(self, text):
        urls = []
        try:
            urls = [url for url in self.url_extractor.gen_urls(text)]
        except Exception as e:
            print(e)
        return urls 

    def url_host_name(self, url):
        empty = ""
        if url.strip() == "":
            return empty
        try:
            if not url.startswith('http'):
                url = 'https://' + url
            usplit = urlsplit(url)
            if usplit.hostname is None:
                # return url
                return empty
            return usplit.hostname
        except Exception as e:
            print("bad url", url,"--",e)
            return empty  # url

    def url_host_names(self, urls):
        hosts = [self.url_host_name(url) for url in urls]
        hosts = [host for host in hosts if host != '']
        return hosts if len(hosts) > 0 else ['']

    def transform(self, tokens, encoder_name):
        assert(self.encoders[encoder_name])
        return self.encoders[encoder_name].transform(tokens)

    def to_matrix(self, var):
        return sparse.csr_matrix(var).reshape(-1, 1)

    # Private methods

    def __clean_string(self, s):
        return re.sub(self.ALPHA_NUMERIC_REGEX, ' ', s)

    def __substitute_num_token(self, s):
        number_token = "#NUMBER#"
        if s.isnumeric() or (s.startswith("-") and s[1:].isnumeric()):
            return number_token
        return s

model_paths = 'model-1598032059/data/'

# Model loader
model = joblib.load(model_paths + "xgb_model/model.pkl")
feature_processor = FeatureProcessor()

encoders = {
    'text': joblib.load(model_paths + "text_encoder/model.pkl"),
    'title': joblib.load(model_paths + "title_encoder/model.pkl"),
    'description': joblib.load(model_paths + "description_encoder/model.pkl"),
    'extension': joblib.load(model_paths + "extension_encoder/model.pkl"),
    'host': joblib.load(model_paths + "host_encoder/model.pkl"),
    'producer': joblib.load(model_paths + "producer_encoder/model.pkl"),
    'checksum': joblib.load(model_paths + "checksum_encoder/model.pkl")
}

feature_processor.set_encoders(encoders)

pred_labels = []
pred_probs = []

for i in range(len(data)):
    raw_input = data[i]
    if not raw_input['description']:
        raw_input['description'] = ''
    feat_vec = feature_processor.feature_vector(
        raw_input['text'],
        raw_input['title'],
        raw_input['description'],
        raw_input['incentivized_upload'],
        raw_input['delta_days'],
        raw_input['is_facebook_user'],
        raw_input['page_count'],
        raw_input['extension'],
        raw_input['producer'],
        raw_input['checksums'],
        raw_input['hyperlinks']
    )

    spam_probability = model.predict_proba(feat_vec)[0][1]

    pred_probs.append(spam_probability)
    pred_labels.append(1 if spam_probability>= 0.5 else 0)


true_labels = []
for i in range(len(data)):
    if data[i]['spam_status'] == 'not_spam':
        true_labels.append(0)
    else:
        true_labels.append(1)

print(precision_recall_fscore_support(true_labels, pred_labels, average='binary'))
