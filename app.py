from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import io

app = Flask(__name__)

# Data & Model Setup
data = pd.read_csv("diabetes.csv", header=None)
if not str(data.iloc[0, 0]).isdigit():
    data = data.iloc[1:]
data = data.apply(pd.to_numeric)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
clf = RandomForestClassifier()
clf.fit(X, y)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    inputs = [float(x) for x in request.form.values()]
    prediction = clf.predict([inputs])
    result = "High Risk of Diabetes" if prediction[0] == 1 else "Low Risk / Normal"
    return render_template("index.html", result=result, inputs=inputs)


@app.route("/download_report", methods=["POST"])
def download_report():
    form_data = request.form
    result = form_data.get("result")

    # PDF Creation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 20, "Diabetes Prediction Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Analysis Result: {result}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, "Patient Data:", ln=True)

    for key, value in form_data.items():
        if key != "result":
            pdf.cell(200, 8, f"- {key.capitalize()}: {value}", ln=True)

    pdf.ln(20)
    pdf.set_text_color(255, 0, 0)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0,
        5,
        "DISCLAIMER: This is an AI-based mathematical model. Please consult a qualified doctor before starting any treatment or medicine.",
    )

    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.cell(200, 10, "Developed by: Kishan Kumar", ln=True, align="R")

    output = io.BytesIO()
    pdf_output = pdf.output(dest="S").encode("latin-1")
    output.write(pdf_output)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="Diabetes_Report.pdf",
        mimetype="application/pdf",
    )


app = app

if __name__ == "__main__":
    app.run(debug=True)
