from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
 
import os
import tensorflow as tf
import numpy as np
import librosa

UPLOAD_FOLDER = 'Buffer'

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

mapping = {
    0 : "Blues",
    1 : "Classical",
    2 : "Country",
    3 : "Disco",
    4 : "Hip_Hop",
    5 : "Jazz",
    6 : "Metal",
    7 : "Pop",
    8 : "Reggae",
    9 : "Rock"
}

model = tf.keras.models.load_model('DL_Model')

app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Songs_db.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SAMPLING_RATE = 22050
n_fft = 2048
hop_length = 512

def extractor(filepath):

    signal, sampling_rate = librosa.load(filepath, sr = SAMPLING_RATE)

    duration = librosa.get_duration(signal)

    num_segments = int(duration / 3) 
    audio_set = 3 * SAMPLING_RATE
    # because our model has been trained on 3 second samples of audio signals

    MFCCs = []
    for s in range(num_segments):

        start_sample = s * audio_set
        end_sample = start_sample + audio_set

        MFCCs.append(librosa.feature.mfcc(signal[start_sample : end_sample],
                                                n_mfcc = 13,
                                                n_fft = n_fft,
                                                hop_length = hop_length))
        
    return MFCCs

# For finding the most frequent element in the list
def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num




# This functions checking if the file name contains a '.' character as well as the file_extension is allowed by the webapp or not
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/filestatus', methods=['GET', 'POST'])
def upload_file():
    # return render_template('upload.html', warning = "I believe you have missed something")
    if request.method == "POST":
        # check if the post request has the file part
        if 'file1' not in request.files:
            return render_template('missing_file.html', warning = "I believe you have missed something")

        file = request.files['file1']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('missing_file.html', warning = "I believe you have missed something")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filename, filepath)
            file.save(filepath)

            # if file.filename.rsplit('.', 1)[-1].lower() == "mp3":
            #     new_name = file.filename.rsplit('.', 1)[0] + ".wav"
            #     print(new_name)

            #     new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
            #     print(new_filepath)
            #     sound = AudioSegment.from_mp3(filepath)
            #     sound.export(new_filepath, format = "wav")

            #     os.remove(filepath)
            #     filepath = new_filepath
                # file.save(new_filepath)
            

            MFCCs = extractor(filepath)
            MFCCs = np.array(MFCCs)
            
            MFCCs = MFCCs[..., np.newaxis]

            labels = []
            rg = MFCCs.shape[0]

            # print(MFCCs, MFCCs.shape, type(MFCCs))
            # print(rg)
            for mfcc in range(rg):

                buff = MFCCs[mfcc]
                buff = buff[np.newaxis, ...]
                pred_val = np.argmax(model.predict(buff))
                labels.append(pred_val)

            ans = most_frequent(labels)
            kart = mapping[ans]

            os.remove(filepath)
            return render_template('result.html', genre = kart)


        else:
            return render_template('weird.html')


# class songs(db.model):
#     sno = db.Column(db.Integer, primary_key = True)
#     song = db.Column()


@app.route("/")
def home():
    return render_template('index.html', name="Kartike Sood")


@app.route("/uploadAudioFile")
def upload():
    return render_template('upload.html')


# @app.route("/extractor", methods=['POST'])
# def extractor():
#     pass


app.run()
# import tensorflow
# print(tensorflow.__version__)
