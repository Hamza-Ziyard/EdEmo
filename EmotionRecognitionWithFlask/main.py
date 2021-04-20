from flask import Flask, render_template, Response, make_response
from camera import VideoCamera

app = Flask(__name__)

video_cam = VideoCamera()



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
    # file_name = video_cam.plot_graph()
    file_name = video_cam.plot_graph()
    # return render_template("results.html", graph="helloooooooo")
    video_cam.__del__()
    return render_template('results.html', graph=file_name)
    # return render_template('results.html')




@app.route('/video_feed')
def video_feed():
    return Response(gen(video_cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
