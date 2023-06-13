from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import googlemaps

app = Flask(__name__)

gmaps = googlemaps.Client(key='AIzaSyDtg5PoT736cvzIpmjvUNAKZ583KWCEdIM')


model = tf.keras.models.load_model('breastfeed_matchmaking_content_based_filtering_model.h5')
scaler = MinMaxScaler()

donor_df = pd.read_csv('donor_breastfeed.csv')

donor_data = donor_df[['Age', 'Dietary Restrictions', 'Religion', 'Health Condition',
       'is_smoke', 'Blood type']].values

k = 5


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    mapped_diet = {'No Restrictions': 0, 'Lactose-free': 1, 'Gluten-free': 2, 'Nut-free': 3, 'Egg-free': 4, 'Vegetarian': 5}
    mapped_health = {'Yes': 1, 'No': 0}
    mapped_religion = {'Muslim': 1, 'Christian': 2, 'Catholic': 3, 'Hinduism': 4, 'Buddhism': 5, 'Konghucu': 6}
    mapped_is_smoke = {'Yes': 1, 'No': 0}
    mapped_blood = {'A': 1, 'B': 2, 'AB': 3, 'O': 4}

    age = int(request.form['Age'])
    diet = mapped_diet.get(request.form['Dietary Restrictions'], None)
    religion = mapped_religion.get(request.form['Religion'], None)
    health = mapped_health.get(request.form['Health Condition'], None)
    is_smoke = mapped_is_smoke.get(request.form['is_smoke'], None)
    blood = mapped_blood.get(request.form['Blood Type'], None)
    location = request.form['Location']

    
    recipient_data = np.array([[age, diet, religion, health, is_smoke, blood]])
    recipient_data_normalized = scaler.fit_transform(recipient_data)

    geocode_result = gmaps.geocode(location)
    if geocode_result:
        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']
        recipient_coordinates = np.array([[lat, lng]])
    else:
        recipient_coordinates = np.array([[0, 0]])  

    recipient_data_normalized = np.concatenate((recipient_data_normalized, recipient_coordinates), axis=1)

    # Normalize the donor data
    donor_data_normalized = scaler.fit_transform(donor_data)

    donor_addresses = donor_df['Location'].to_list()
    donor_coordinates = []
    for address in donor_addresses:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            lat = geocode_result[0]['geometry']['location']['lat']
            lng = geocode_result[0]['geometry']['location']['lng']
            donor_coordinates.append([lat, lng])
        else:
            donor_coordinates.append([0, 0])  
    donor_coordinates = np.array(donor_coordinates)

    donor_data_normalized = np.concatenate((donor_data_normalized, donor_coordinates), axis=1)

    recipient_representation = model.predict(recipient_data_normalized)
    similarity = np.linalg.norm(donor_data_normalized - recipient_representation, axis=1)
    top_k_indices = np.argsort(similarity)[:k]
    top_k_similar_donors = donor_df.iloc[top_k_indices]

    return render_template('recommendations.html', donors=top_k_similar_donors)


if __name__ == '__main__':
    app.run()
