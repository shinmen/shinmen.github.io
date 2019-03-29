
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

var ui = ui || {}


const CONTROLS = [
  'up', 'down', 'left', 'right', 'leftclick', 'rightclick', 'scrollup',
  'scrolldown'
];
var controlsCaptured = [], labelsCaptured = [];

ui.init =
    function() {
  document.getElementById('user-help').style.visibility = 'visible';
  document.getElementById('user-help-text').innerText =
      'Add images to the classes below by clicking or holding';
  var controlButtons = document.getElementsByClassName('control-button');
  for (var i = 0; i < controlButtons.length; i++) {
    controlButtons[i].addEventListener('mouseover', function(event) {
      if (event.target.classList.contains('control-button')) {
        document.getElementById(event.target.id + '-icon').className =
            'control-icon center move-up';
        document.getElementById(event.target.id + '-add-icon').className =
            'add-icon';
      }
    });
    controlButtons[i].addEventListener('mouseout', function(event) {
      if (event.target.classList.contains('control-button')) {
        document.getElementById(event.target.id + '-icon').className =
            'control-icon center';
        document.getElementById(event.target.id + '-add-icon').className =
            'add-icon invisible';
      }
    });
  }
}



function
hideAllDropdowns() {
  let dropdownLists = document.getElementsByClassName('custom-option-list');
  for (var j = 0; j < dropdownLists.length; j++) {
    dropdownLists[j].className = 'custom-option-list hide';
  }
}

var customDropdowns = document.getElementsByClassName('custom-dropdown');
for (var i = 0; i < customDropdowns.length; i++) {
  customDropdowns[i].addEventListener('click', (event) => {
    hideAllDropdowns();
    const id = event.target.id + '-list';
    document.getElementById(id).className = 'custom-option-list';
  });
}

var customDropdownOptions = document.getElementsByClassName('custom-option');
for (var i = 0; i < customDropdownOptions.length; i++) {
  customDropdownOptions[i].addEventListener('click', (event) => {
    let dropdownID = event.target.parentNode.getAttribute('dropdownID');
    let dropdownList = document.getElementById(dropdownID + '-dropdown-list');
    dropdownList.getElementsByClassName('selected')[0].className =
        'custom-option';
    event.target.className = 'custom-option selected';
    event.target.parentNode.className = 'custom-option-list hide';
    document.getElementById(dropdownID).innerText = event.target.innerText;
  });
}

document.body.addEventListener('click', (event) => {
  if (event.target.className !== 'custom-dropdown') {
    hideAllDropdowns();
  }
})

const trainStatusElement = document.getElementById('train-status');
const downloadModel = document.getElementById('download-model');

// Set hyper params from UI values.
const learningRateElement = document.getElementById('learningRate');
var selectedLearningRateValue =
    document.getElementById('learningRate-dropdown-list')
        .getElementsByClassName('selected')[0]
        .innerText;
learningRateElement.innerText = selectedLearningRateValue;

ui.getLearningRate = () => +learningRateElement.innerText;

const batchSizeFractionElement = document.getElementById('batchSizeFraction');
var batchSizeFractionValue =
    document.getElementById('batchSizeFraction-dropdown-list')
        .getElementsByClassName('selected')[0]
        .innerText;
batchSizeFractionElement.innerText = batchSizeFractionValue;

ui.getBatchSizeFraction = () => +batchSizeFractionElement.innerText;

const epochsElement = document.getElementById('epochs');
var epochsValue = document.getElementById('epochs-dropdown-list')
                      .getElementsByClassName('selected')[0]
                      .innerText;
epochsElement.innerText = epochsValue;

ui.getEpochs = () => +epochsElement.innerText;

const denseUnitsElement = document.getElementById('dense-units');
var denseUnitsValue = document.getElementById('dense-units-dropdown-list')
                          .getElementsByClassName('selected')[0]
                          .innerText;
denseUnitsElement.innerText = denseUnitsValue;

ui.getDenseUnits = () => +denseUnitsElement.innerText;

function removeActiveClass() {
  let activeElement = document.getElementsByClassName('active');
  while (activeElement.length > 0) {
    activeElement[0].className = 'control-inner-wrapper';
  }
}

ui.predictClass =
    function(classId) {
  removeActiveClass();
  classId = Math.floor(classId);
  document.getElementById(controlsCaptured[classId] + '-button').className =
      'control-inner-wrapper active';
  document.body.setAttribute('data-active', controlsCaptured[classId]);
}

    ui.isPredicting =
        function() {
  document.getElementById('predict').className = 'test-button hide';
  document.getElementById('webcam-outer-wrapper').style.border =
      '4px solid #00db8b';
  document.getElementById('stop-predict').className = 'stop-button';
  document.getElementById('bottom-section').style.pointerEvents = 'none';
  downloadModel.className = 'disabled';
} 
ui.donePredicting =
            function() {
  document.getElementById('predict').className = 'test-button';
  document.getElementById('webcam-outer-wrapper').style.border =
      '2px solid #c8d0d8';
  document.getElementById('stop-predict').className = 'stop-button hide';
  document.getElementById('bottom-section').style.pointerEvents = 'all';
  downloadModel.className = '';
  removeActiveClass();
}

            ui.trainStatus =
                function(status) {
  trainStatusElement.innerText = status;
}

                ui.enableModelDownload =
                    function() {
  downloadModel.className = '';
}

                    ui.enableModelTest =
                        function() {
  document.getElementById('predict').className = 'test-button';
}

var addExampleHandler;

ui.setExampleHandler = function(handler) {
  addExampleHandler = handler;
} 
let mouseDown = false;
const totals = [0, 0, 0, 0, 0, 0, 0, 0];

const upButton = document.getElementById('up');
const downButton = document.getElementById('down');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');
const leftClickButton = document.getElementById('leftclick');
const rightClickButton = document.getElementById('rightclick');
const scrollUpButton = document.getElementById('scrollup');
const scrollDownButton = document.getElementById('scrolldown');

const thumbDisplayed = {};
function timeout(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
ui.handler =
    async function(label) {
  mouseDown = true;
  const id = CONTROLS[label];
  const button = document.getElementById(id);
  const total = document.getElementById(id + '-total');

  document.body.removeAttribute('data-active');


  while (mouseDown) {
    totals[label] = totals[label] + 1;
    total.innerText = totals[label];
    for (var i = 0; i < totals.length; i++) {
      if (totals[i] > 0) {
        var isPresent = false;
        for (var j = 0; j < controlsCaptured.length; j++) {
          if (CONTROLS[i] === controlsCaptured[j]) {
            isPresent = true;
            break;
          }
        }
        if (!isPresent) {
          controlsCaptured.push(CONTROLS[i]);
          labelsCaptured.push(i);
          break;
        }
      }
    }
    addExampleHandler(label);
    await Promise.all([tf.nextFrame(), timeout(300)]);
  }
  document.body.setAttribute('data-active', CONTROLS[label]);
  if (controlsCaptured.length >= 2) {
    document.getElementById('train').className = 'train-button';
    ui.trainStatus('TRAIN');
    document.getElementById('predict').className = 'test-button disabled';
    downloadModel.className = 'disabled';
    document.getElementById('user-help-text').innerText =
        'Add more images or train the model';
  } else {
    document.getElementById('user-help-text').innerText =
        'Minimum of 2 classes required to train the model';
  }
}

    upButton.addEventListener('mousedown', () => ui.handler(0));
upButton.addEventListener('mouseup', () => {
  mouseDown = false;
});

downButton.addEventListener('mousedown', () => ui.handler(1));
downButton.addEventListener('mouseup', () => mouseDown = false);

leftButton.addEventListener('mousedown', () => ui.handler(2));
leftButton.addEventListener('mouseup', () => mouseDown = false);

rightButton.addEventListener('mousedown', () => ui.handler(3));
rightButton.addEventListener('mouseup', () => mouseDown = false);

leftClickButton.addEventListener('mousedown', () => ui.handler(4));
leftClickButton.addEventListener('mouseup', () => mouseDown = false);

rightClickButton.addEventListener('mousedown', () => ui.handler(5));
rightClickButton.addEventListener('mouseup', () => mouseDown = false);

scrollUpButton.addEventListener('mousedown', () => ui.handler(6));
scrollUpButton.addEventListener('mouseup', () => mouseDown = false);

scrollDownButton.addEventListener('mousedown', () => ui.handler(7));
scrollDownButton.addEventListener('mouseup', () => mouseDown = false);

ui.drawThumb =
    function(img, label) {
  if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
    thumbCanvas.style.display = 'block';
    document.getElementById(CONTROLS[label] + '-icon').style.top = '-50%';
    ui.draw(img, thumbCanvas);
  }
}

    ui.draw = function(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}


// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// The number of classes we want to predict.

var NUM_CLASSES = 8;

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let mobilenet;
let model;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the button. up, down, left, right are
// labels 0, 1, 2, 3 respectively.
ui.setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    controllerDataset.addExample(mobilenet.predict(img), label);

    NUM_CLASSES = totals.filter(total => total > 0).length;

    // Draw the preview thumbnail.
    ui.drawThumb(img, label);
  });
  ui.trainStatus('TRAIN');
});

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      // Layer 1
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: tf.initializers.varianceScaling(
            {scale: 1.0, mode: 'fanIn', distribution: 'normal'}),
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: tf.initializers.varianceScaling(
            {scale: 1.0, mode: 'fanIn', distribution: 'normal'}),
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(ui.getLearningRate());
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }
  let loss = 0;
  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        document.getElementById('train').className =
            'train-model-button train-status';
        loss = logs.loss.toFixed(5);
        ui.trainStatus('LOSS: ' + logs.loss.toFixed(5));
      },
      onTrainEnd: () => {
        if (loss > 1) {
          document.getElementById('user-help-text').innerText =
              'Model is not trained well. Add more samples and train again';
        } else {
          document.getElementById('user-help-text').innerText =
              'Test or download the model. You can even add images for other classes and train again.';
          ui.enableModelTest();
          ui.enableModelDownload();
        }
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();

      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model.
      const activation = mobilenet.predict(img);

      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.
      const predictions = model.predict(activation);

      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('ENCODING...');
  controllerDataset.ys = null;
  controllerDataset.addLabels(NUM_CLASSES);
  ui.trainStatus('TRAINING...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  isPredicting = true;
  predict();
});

document.getElementById('stop-predict').addEventListener('click', () => {
  isPredicting = false;
  predict();
});

async function init() {
  try {
    await webcam.setup();
    console.log('Webcam is on');
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
    document.getElementById('webcam-inner-wrapper').className =
        'webcam-inner-wrapper center grey-bg';
    document.getElementById('bottom-section').style.pointerEvents = 'none';
  }

  mobilenet = await loadMobilenet();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  ui.init();
}

var a = document.createElement('a');
document.body.appendChild(a);
a.style = 'display: none';

document.getElementById('download-model').onclick =
    async () => {
  await model.save('downloads://model');

  var text = controlsCaptured.join(',');
  var blob = new Blob([text], {type: 'text/csv;charset=utf-8'});
  var url = window.URL.createObjectURL(blob);
  a.href = url;
  a.download = 'labels.txt';
  a.click();
  window.URL.revokeObjectURL(url);
}

// Initialize the application.
init();
