let net;

const CL_UNCERTAIN = 0.0;
const CL_LEAST_CERTAINTY = 0.2;
const CL_LOW_CERTAINTY = 0.4;
const CL_MEDIUM_CERTAINTY = 0.6;
const CL_HIGH_CERTAINTY = 0.8;
const CL_SUPER_CERTAIN = 1.0;


const CL_UNCERTAIN_STR = "UNCERTAIN";
const CL_LEAST_CERTAINTY_STR = "LEAST_CERTAINTY";
const CL_LOW_CERTAINTY_STR = "LOW_CERTAINTY";
const CL_MEDIUM_CERTAINTY_STR = "MEDIUM_CERTAINTY"; 
const CL_HIGH_CERTAINTY_STR = "HIGH_CERTAINTY";
const CL_SUPER_CERTAIN_STR = "SUPER_CERTAIN";


// This function analyzes and reports the certainity of the prediction. 
// The range for different levels of certainty is discribed by the constants 
// defined above, prefixed with `CL_`
function report_certainty (top_prob) {
	var certainty = document.getElementById('certainty'); 
	var report = "";
	if (top_prob == CL_UNCERTAIN){
		report = CL_UNCERTAIN_STR; 
	} else if (CL_UNCERTAIN < top_prob &&  top_prob <= CL_LEAST_CERTAINTY) {
		report = CL_LEAST_CERTAINTY_STR;
	} else if (CL_LEAST_CERTAINTY < top_prob && top_prob <= CL_LOW_CERTAINTY) {
		report = CL_LOW_CERTAINTY_STR;
	} else if (CL_LOW_CERTAINTY < top_prob && top_prob <= CL_MEDIUM_CERTAINTY) {
		report = CL_MEDIUM_CERTAINTY_STR;
	} else if (CL_MEDIUM_CERTAINTY < top_prob && top_prob <= CL_HIGH_CERTAINTY){
		report = CL_HIGH_CERTAINTY_STR;
	} else if (CL_HIGH_CERTAINTY < top_prob && top_prob <= CL_SUPER_CERTAIN) {
		report = CL_SUPER_CERTAIN_STR;
	} 
	
	// unless there is some kind of an error, there must always be a report displayed. 
	certainty.innerText = `PREDICTION CERTAINTY: ${report}`
}	

const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

async function app() {
  var no_action_imgs_active = true;
  var num_of_no_action_imgs = 0;
  var active_classes = 0;
  
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  await setupWebcam();
  
  var start_time = new Date();
  var end_time = new Date();
  // var diff = date2 - date1; //milliseconds interval

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = classId => {
    
    // we must check for active classes because, the classes must be trained in their 
    // numerical order. For example, the first class must be trained first, the 
    // second class must be trained second and so on. Not following this order leads
    // to uncertain behavior. 
    if (active_classes >= classId ){
	    // reset start-time on this event.
	    start_time = new Date()
	    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
	    // to the KNN classifier.
	    const activation = net.infer(webcamElement, 'conv_preds');

	    // Pass the intermediate activation to the classifier.
	    classifier.addExample(activation, classId);
	    if (active_classes == classId){
		active_classes += 1;
	    }
    }
  };


  // Starts/stops reading no-action images.
  const off_on_noaction = () => {
    if (no_action_imgs_active){
	no_action_imgs_active = false;
    } else {
	no_action_imgs_active = true;
    }
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));
  document.getElementById('class-d').addEventListener('click', () => addExample(3));
  document.getElementById('on-of-noaction').addEventListener('click', () => off_on_noaction());

  while (true) {
    if (classifier.getNumClasses() > 0) {
      
	// Get the activation from mobilenet from the webcam.
	const activation = net.infer(webcamElement, 'conv_preds');
	// Get the most likely class and confidences from the classifier module.
	     
	// Here, K=20 is set. Greater the number of classifiers, greater the prediction certainity. Also 
	// it would be better if it is a multiple of the number of classes (excluding the 'No Action' class). 
	const result = await classifier.predictClass(activation, 32);

	const classes = ['A', 'B', 'C', 'D', 'No Action'];
	var console_div = document.getElementById('console');
	var noaction_report = document.getElementById('adding-image');
	console_div.innerText = `Top Prediction: ${classes[result.classIndex]}\n`+
	    		      `Top prediction's probability: ${result.confidences[result.classIndex]}\n`;

	for (var i = 0; i < classes.length; i++) {
		if (typeof result.confidences[i] !== 'undefined'){
			console_div.innerText+= `Class ${classes[i]}: ${result.confidences[i]}\n`;			
		}
	}   	
	report_certainty(result.confidences[result.classIndex]);
	
	noaction_report.innerText = `\n\n\n Number of no-action images: ${num_of_no_action_imgs}`;
	noaction_report.innerText += `\n Is the system actively adding No-action images? ${((no_action_imgs_active)?"Yes":"No")}`;
	

	end_time = new Date();
	if ( no_action_imgs_active && (end_time - start_time > 1000) &&(active_classes > 3) ){
		start_time = new Date();
		// Get the intermediate activation of MobileNet 'conv_preds' and pass that
	    	// to the KNN classifier.
	    	const activation = net.infer(webcamElement, 'conv_preds');
		classifier.addExample(activation, 4);
		num_of_no_action_imgs += 1;
	} 
    }
    await tf.nextFrame();
  }
}

app();
