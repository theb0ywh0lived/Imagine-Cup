import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import dill as pickle
from sklearn.feature_extraction.text import HashingVectorizer
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    try:
        test_json=request.get_json()
        test=pd.read_json(test_json,orient='records')
        test['title']=[str(x) for x in list(test['title'])]
        test['text']=[str(x) for x in list(test['text'])]
        test['label']=[str(x) for x in list(test['label'])]
        hash_vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
        hash_test = hash_vectorizer.transform(test)
        

    except Exception as e:
        raise e
    clf='model_v1.pk'
    if(test.empty):
        return (bad_request())
    
    else:
            
        print("Loading the model...")
        loaded_model = None
        with open(clf,'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(hash_test)
        prediction_series = list(pd.Series(predictions))
        final_predictions = pd.DataFrame(list(zip( prediction_series)))
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)

if __name__=='__main__':
	app.run()
