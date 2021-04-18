const video = document.getElementById('video')

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models')
]).then(startVideo)

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )
}

video.addEventListener('play', () => {
  const canvas = faceapi.createCanvasFromMedia(video)
  document.body.append(canvas)
  const displaySize = { width: video.width, height: video.height }
  faceapi.matchDimensions(canvas, displaySize)
  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    let chart = canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    // faceapi.draw.drawDetections(canvas, resizedDetections)
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections)

    
  }, 100)
})


// let chart = document.getElementById('myChart').getContext('2d');

//     let lineChart = new Chart(chart, {
//       type: 'line',
//       data: {
//         labels: ['Happy','Sad','Neutral','Surprised','Angry','Fear','Disgust'],
//         datasets: [{
//           label: 'Emotion',
//           data:[
//              1000,
//              1120,
//              200,
//              400,
//              1000,
//              1100,
//              300
//           ]
//         }]
//       },
//       options: {}
//     });

// var engagedDetection =['Happy','Neutral','Surprised']
// var disengagedDetection =['Sad','Fear','Angry','Disgust']

// var getEmotion = ['Engaged','Disengaged']

// for (var i=0; i<engagedDetection.length; i++){
//   console.log('Engaged '+engagedDetection[i])
// }

// for (var i=0; i<disengagedDetection.length; i++){
//   console.log('Dis-Engaged '+disengagedDetection[i])
// }


// let chart = document.getElementById('myChart').getContext('2d');

//     let lineChart = new Chart(chart, {
//       type: 'line',
//       data: {
//         labels: detections,
//         datasets: [{
//           label: 'Emotion',
//           data:[
//              1000,
//              1120,
//              200,400
//           ]
//         }]
//       },
//       options: {}
//     });