# print("Hello")
from gravityai import gravityai as grav
import pickle
import pandas as pd

model = pickle.load(open('financial_text_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('financial_text_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('financial_text_encoder.pkl', 'rb'))

def process(inPath, outPath):
    # read input file
    input_df = pd.read_csv(inPath)
    # vectorize the data
    features = tfidf_vectorizer.transform(input_df['body'])
    # predict classses
    predictions = model.predict(features)
    # convert predictions to labels
    labels = label_encoder.inverse_transform(predictions)
    # add labels to the input dataframe
    input_df['category'] = labels
    # write output file
    output_df = input_df[['id', 'category']]
    output_df.to_csv(outPath, index=False)

grav.wait_for_request(process)