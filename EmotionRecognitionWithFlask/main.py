from flask import Flask, render_template, Response, make_response, request, session
from camera import VideoCamera
from flask_mail import Mail, Message
from flask_session import Session
from datetime import date,datetime


app = Flask(__name__)

video_cam = VideoCamera()
app.secret_key = "wCazB6p1HZ"

@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/results')
def display_graph():
    file_name=video_cam.plot_graph()
    session['results_image_name'] = file_name
    # return render_template("results.html", graph="helloooooooo")
    video_cam.__del__()
    return render_template('results.html', graph=file_name)
    # return render_template('results.html')




@app.route('/video_feed')
def video_feed():
    return Response(gen(video_cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


email_id = 'edemodc@gmail.com'
email_pw = 'demolitioncrew'

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = email_id
app.config['MAIL_PASSWORD'] = email_pw
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)


@app.route('/send_message', methods=['GET', 'POST'])
def send_result():
    if request.method == "POST":
        email = request.form['email']
        subject = "Engagement results from EdEmo"
        participant_name = request.form['name']

        today = date.today()
        now_date = today.strftime("%B %d, %Y")
        now = datetime.now()
        now_time = now.strftime("%H:%M:%S")


        msg = "Participant name:  " + participant_name + "\nDate: " + now_date + "\nTime: " + now_time
        message = Message(subject, sender="edemodc@gmail.com", recipients=[email])
        message.body = msg
        file_name = session['results_image_name']

        with app.open_resource("static\\"+file_name) as fp:
            message.attach(file_name, "image/png", fp.read())

        mail.send(message)
        success = "Successfully Sent"
        return render_template("success.html", success=success)
    else:
        return render_template('results.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
