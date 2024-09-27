import os
from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import pytesseract

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'your_secret_key'

# Set the path for Tesseract-OCR if needed (only for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 200)
    return image, edged

def detect_plate(image, edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate = image[y:y + h, x:x + w]
            return plate
    return None

def recognize_plate(plate):
    gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh_plate, config='--psm 8')
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image, edged = preprocess_image(file_path)
            plate = detect_plate(image, edged)

            if plate is not None:
                plate_text = recognize_plate(plate)
                return render_template("index.html", plate_text=plate_text)

            flash("No number plate detected.")
            return redirect(request.url)
    return render_template("index.html")

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
