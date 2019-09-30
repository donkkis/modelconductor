# Electric Motor Temperature
# 140 hrs recordings from a permanent magnet synchronous motor (PMSM)
# Data source : https://www.kaggle.com/wkirgsn/electric-motor-temperature
import requests
import pandas as pd
import os
import sys
from clint.textui import progress

def prepare_dataset():
    # Get and prepare raw data
    # Display the progressbar in std output
    progress.STREAM = sys.stdout
    print("Downloading test files (130 MB)")
    url = 'https://www.dropbox.com/s/s88h11rpn4x86fo/pmsm_temperature_data.csv?dl=1'
    out_path = '..\\testresources\\pmsm_temperature_data.csv'
    r = requests.get(url, stream=True)
    if not os.path.exists(out_path):
        with open(out_path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            it = r.iter_content(chunk_size=1024)
            for chunk in progress.bar(it, expected_size=(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()

    data = pd.read_csv(out_path)
    data = data.drop('profile_id', axis=1)

    # Fit some arbitrary model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(normalize=True)

    # Split data to independent / dependent variables
    # y = Permanent Magnet surface temperature representing the rotor temperature.
    from sklearn.model_selection import train_test_split
    X = data.drop('pm', axis=1)
    y = data['pm']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)
    # Fit the model
    model.fit(X_train, y_train)

    # Export model
    model_out_path = '..\\testresources\\pmsm.pickle'
    if not os.path.exists(model_out_path):
        import pickle
        with open(model_out_path, 'wb') as f:
            pickle.dump(model, f)

    # Get a small sample for testing purposes
    sample_out_path = '..\\testresources\\pmsm_temperature_sample.csv'
    if not os.path.exists(sample_out_path):
        data_sample = data.sample(n=1000, random_state=42)
        data_sample.to_csv('..\\testresources\\pmsm_temperature_sample.csv',
                           index=False)


if __name__ == '__main__':
    prepare_dataset()
