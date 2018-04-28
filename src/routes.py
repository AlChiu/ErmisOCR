"""routes.py
Main routes to define the paths the Flask server.
"""
import os
import glob
import cv2
from flask import flash, render_template, request, redirect, url_for
from src import app
from werkzeug.utils import secure_filename
from src.classifier import classifier
from src.detector import char_detect_segment as det_seg

# Load the classifier
M_PTH = "src/classifier/model62_ensemble.hdf5"
L_PTH = "src/classifier/char_labels62.json"
CLASSIFIER = classifier.Classifier(M_PTH, L_PTH)

# Image upload configuration
UPLOAD_FOLDER = 'src/static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret'


def allowed_file(filename):
    """
    DESCRIPTION: Check if file has the correct extension
    INPUT: Full filename
    OUTPUT: Boolean if the file is allowed
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Homepage/Index Route
@app.route('/')
@app.route('/index')
def index():
    """
    DESCRIPTION: Render the upload/homepage
    """
    return render_template('index.html', title='Home')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    DESCRIPTION: Uploads an image, server will segment and classify
    INPUT: Full image file
    OUTPUT: Translation of image file
    """
    # Path to the segmented word and character images.
    pth = "src/classifier/segmented/"

    # List of words that have been classified
    c_words = []

    # User uploads a valid image
    if request.method == 'POST':
        # Check if there is a file in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))
        # If there is a file part
        up_file = request.files['file']
        # Check if the user submitted no file
        if up_file.filename == '':
            flash('No file selected!')
            return redirect(url_for('index'))
        # If there is a file, check if it is a valid file
        if up_file and allowed_file(up_file.filename):
            filename = secure_filename(up_file.filename)
            save_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            up_file.save(save_file)
        else:
            flash('Invalid file!')
            return redirect(url_for('index'))

        # Detect and segment words
        words = det_seg.detector_for_words(save_file)
        det_seg.segment_words(words.Image,
                              words.Boxes,
                              pth)

        # For each word, detect, segment, and classify
        # the characters in that particular word.
        for word in sorted(glob.iglob(pth + 'word_*.png'),
                           key=det_seg.numerical_sort):
            chars = det_seg.detector_for_characters(word)
            w_image = cv2.imread(word)
            det_seg.segment_chars(chars.Name,
                                  w_image,
                                  chars.Boxes,
                                  pth)
            # Classify the sorted, segmented characters of each word
            c_words.append(CLASSIFIER.classify_many(pth))
            os.remove(word)

        # Render the translation on the show_entries page
        return render_template('show_entries.html',
                               data=c_words,
                               title="Translation",
                               upfile=save_file,
                               filename=up_file.filename)
