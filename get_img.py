#Script para descargar imagenes de la web.

#Importar librerias
import time
from datetime import datetime
import urllib.request

#Array de webcambs
webcam = [
("Torrington","http://67.43.220.114:80/jpg/image.jpg?"),
("Montreal","http://70.81.224.78:80/webcapture.jpg?command=snap&channel=1?"),
("Iga","http://111.64.36.153:50001/cgi-bin/camera?resolution=640&amp;quality=1&amp;Language=0&amp;"),
("Asahi","http://124.155.121.218:3000/webcapture.jpg?command=snap&channel=1?"),
("Kwangmyong","http://121.125.133.92:8000/webcapture.jpg?command=snap&channel=1?"),
("Mobile","http://170.249.152.2:8080/cgi-bin/viewer/video.jpg?r=")
]

#Generar nombre archivo unico
d = datetime.now()
dstr = "_"+str(d.year)+"-"+str(d.month)+"-"+str(d.day)+"_"+str(d.hour)+"h"+str(d.minute)+"m.jpg"
path = "/home/pi/Pictures/parking/"

#Bucle para descargar imagenes del array
for (city,url) in webcam:
    #Descargar imagen
    urllib.request.urlretrieve(url+str(int(time.time())), path+city+dstr)
    #print("Downloaded "+city)
#print("Finished downloading")

import os
import pandas as pd
import json

# Set directory containing JSON files
json_dir = 'images/train'

# Which set to assign the contents in the rows to
set_type = 'TRAIN' # See https://cloud.google.com/vision/automl/object-detection/docs/csv-format

# Initialize empty DataFrame to store data
df_all = pd.DataFrame()

# Wrapper
def read_transform(json_dir, set_type):
    global df_all
    # Loop over JSON files in directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            # Load JSON file
            with open(os.path.join(json_dir, filename)) as f:
                data = json.load(f)
            # Extract relevant fields
            #print(data)
            imagePath = json_dir+'/'+data['imagePath']
            imageHeight = data['imageHeight']
            imageWidth = data['imageWidth']
            res = [{'set_type': set_type, 'imagePath': imagePath, 'label': shape['label'], 'p1x': shape['points'][0][0]/imageWidth, 'p1y': shape['points'][0][1]/imageHeight, 'p2x': '', 'p2y': '', 'p3x': shape['points'][1][0]/imageWidth, 'p3y': shape['points'][1][1]/imageHeight, 'p4x': '', 'p4y': ''} for shape in data['shapes']]
            #print(res[0])
            # Convert JSON to DataFrame
            df = pd.json_normalize(res)
            # Concatenate DataFrame to df_all
            df_all = pd.concat([df_all, df], ignore_index=True)

# For training
read_transform(json_dir, set_type)
# For validation
json_dir = 'images/validation'
set_type = 'VALIDATION'
read_transform(json_dir, set_type)
# Write concatenated DataFrame to CSV file
json_dir = 'images/'
df_all.to_csv(os.path.join(json_dir, 'all_data.csv'), index=False, header=False)
print('done writting '+str(df_all.shape))

train_data, validation_data, _ = object_detector.DataLoader.from_csv('images/all_data.csv')

model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)

model.export(export_dir='.')


python3 -m venv env
source env/bin/activate

pip install -r requirements.txt


@app.route('/')
def root():
    # This loads the home page.
    return render_template('index.html')

@app.route('/processing', methods=['GET'])
def process():
    # Retrieve the arguments from the call. In this case the name of the parking
    args = request.args.to_dict()
	# Inference function is called with the parmeter to get the numebr of cars detected.
    data, cars = run_odt_and_draw_results(args.get("name"), threshold=0.3)
	# Load the results page
    return render_template('result.html', data=data, cars=cars)




def run_odt_and_draw_results(parking_name, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the model and its configuration
  interpreter = tf.lite.Interpreter(model_path=model_path)

  # Retrieve the image of the specified parking in real time
  parking_image = preprocess_image(parking_name)

  # Run object detection on the input image
  results = detect_objects(parking_name, interpreter, threshold=threshold)

  # Plot the detection results on the input image
  cars_detected_image, cars_detected_count = interpret_results(results)

  return cars_detected_image, cars_detected_coun


python main.py

gcloud app deploy

gcloud app browse
