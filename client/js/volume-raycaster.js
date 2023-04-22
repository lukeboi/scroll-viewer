var cubeStrip = [
	1, 1, 0,
	0, 1, 0,
	1, 1, 1,
	0, 1, 1,
	0, 0, 1,
	0, 1, 0,
	0, 0, 0,
	1, 1, 0,
	1, 0, 0,
	1, 1, 1,
	1, 0, 1,
	0, 0, 1,
	1, 0, 0,
	0, 0, 0
];

var takeScreenShot = false;
var canvas = null;

var gl = null;
var shader = null;
var volumeTexture = null;
var colormapTex = null;
var fileRegex = /.*\/(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;
var proj = null;
var camera = null;
var projView = null;
var tabFocused = true;
var newVolumeUpload = true;
var targetFrameTime = 32;
var samplingRate = 1.0;
var near_clip = 0.1;
var far_clip = 100.0;
var zoom_increment = 1;
var WIDTH = 0;
var HEIGHT = 0;
var bboxMin = null;
var bboxMax = null;
var volDims = [560, 560, 477]

var backgroundColor = [0.0, 0.0, 0.0]

const defaultEye = vec3.set(vec3.create(), 0.5, 0.5, 1.5);
const center = vec3.set(vec3.create(), 0.5, 0.5, 0.5);
const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);

var volumes = {
	"Fuel": "7d87jcsh0qodk78/fuel_64x64x64_uint8.raw",
	"Neghip": "zgocya7h33nltu9/neghip_64x64x64_uint8.raw",
	"Hydrogen Atom": "jwbav8s3wmmxd5x/hydrogen_atom_128x128x128_uint8.raw",
	"Boston Teapot": "w4y88hlf2nbduiv/boston_teapot_256x256x178_uint8.raw",
	"Engine": "ld2sqwwd3vaq4zf/engine_256x256x128_uint8.raw",
	"Bonsai": "rdnhdxmxtfxe0sa/bonsai_256x256x256_uint8.raw",
	"Foot": "ic0mik3qv4vqacm/foot_256x256x256_uint8.raw",
	"Skull": "5rfjobn0lvb7tmo/skull_256x256x256_uint8.raw",
	"Aneurysm": "3ykigaiym8uiwbp/aneurism_256x256x256_uint8.raw",
};

var colormaps = {
	"Eat your greens": "colormaps/samsel-linear-green.png",
	"Black to white (CT Scan)": "colormaps/black-to-white.png",
	"Banana": "colormaps/yellow-parched.png",
	"Firey Ice": "colormaps/cool-warm-paraview.png",
	"Purple Rain": "colormaps/matplotlib-plasma.png",
	"Deep Blue": "colormaps/matplotlib-virdis.png",
	"Iridesence (2018)": "colormaps/rainbow.png",
	"Green inverted": "colormaps/samsel-linear-ygb-1211g.png",
};

var loadVolume = function(file, onload) {
	// var url = "/output.raw";
	var url = "http://localhost:5000/random_volume?size=560,560,477"
	var req = new XMLHttpRequest();
	var loadingProgressText = document.getElementById("loadingText");
	var loadingProgressBar = document.getElementById("loadingProgressBar");

	loadingProgressText.innerHTML = "Loading Volume";
	loadingProgressBar.setAttribute("style", "width: 0%");

	req.open("GET", url, true);
	req.responseType = "arraybuffer";
	req.onprogress = function(evt) {
		var vol_size = volDims[0] * volDims[1] * volDims[2];
		var percent = evt.loaded / vol_size * 100;
		loadingProgressBar.setAttribute("style", "width: " + percent.toFixed(2) + "%");
	};
	req.onerror = function(evt) {
		loadingProgressText.innerHTML = "Error Loading Volume";
		loadingProgressBar.setAttribute("style", "width: 0%");
	};
	req.onload = function(evt) {
		loadingProgressText.innerHTML = "Loaded Volume";
		loadingProgressBar.setAttribute("style", "width: 100%");
		var dataBuffer = req.response;
		if (dataBuffer) {
			dataBuffer = new Uint8Array(dataBuffer);
			onload(file, dataBuffer);
		} else {
			alert("Unable to load buffer properly from volume?");
			console.log("no buffer?");
		}
	};
	req.send();
}

var selectVolume = function() {
	loadVolume(volumes[0],function(file, dataBuffer) {
		var renderTimeText = document.getElementById("fpsText");
		var volumeSizeText = document.getElementById("volumeSizeText");
		
		bboxMin = [0.0, 0.0, 0.0]
		bboxMax = [1.0, 1.0, 1.0]

		var byteSize = volDims[0] * volDims[1] * volDims[2];
		volumeSizeText.innerHTML = "Volume Dimensions: " + volDims.join("x") + " (" + Math.round(byteSize / 1000000 ) + "mb)";

		updateProjectionMatrix(null);

		var tex = gl.createTexture();
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_3D, tex);
		gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R8, volDims[0], volDims[1], volDims[2]);
		gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0,
			volDims[0], volDims[1], volDims[2],
			gl.RED, gl.UNSIGNED_BYTE, dataBuffer);

		var longestAxis = Math.max(volDims[0], Math.max(volDims[1], volDims[2]));
		var volScale = [volDims[0] / longestAxis, volDims[1] / longestAxis,
			volDims[2] / longestAxis];

		gl.uniform3iv(shader.uniforms["volume_dims"], volDims);
		gl.uniform3fv(shader.uniforms["volume_scale"], volScale);

		newVolumeUpload = true;
		if (!volumeTexture) {
			volumeTexture = tex;
			setInterval(function() {
				// Save them some battery if they're not viewing the tab
				if (document.hidden) {
					return;
				}
				var startTime = performance.now();
				gl.clearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0);
				gl.clear(gl.COLOR_BUFFER_BIT);

				// Reset the sampling rate and camera for new volumes
				if (newVolumeUpload) {
					camera = new ArcballCamera(defaultEye, center, up, zoom_increment, [WIDTH, HEIGHT]);
					samplingRate = 1.0;
					gl.uniform1f(shader.uniforms["dt_scale"], samplingRate);
				}
				gl.uniform1f(shader.uniforms["near_clip"], near_clip);
				gl.uniform1f(shader.uniforms["far_clip"], far_clip);
				projView = mat4.mul(projView, proj, camera.camera);
				gl.uniformMatrix4fv(shader.uniforms["proj_view"], false, projView);

				gl.uniform3fv(shader.uniforms["new_box_min"], bboxMin);
				gl.uniform3fv(shader.uniforms["new_box_max"], bboxMax);

				var eye = [camera.invCamera[12], camera.invCamera[13], camera.invCamera[14]];
				gl.uniform3fv(shader.uniforms["eye_pos"], eye);

				gl.drawArrays(gl.TRIANGLE_STRIP, 0, cubeStrip.length / 3);
				// Wait for rendering to actually finish
				gl.finish();
				var endTime = performance.now();
				var renderTime = endTime - startTime;

				// Update render time
				renderTimeText.innerHTML = "FPS: " + Math.round(1000 / renderTime);

				var targetSamplingRate = renderTime / targetFrameTime;

				if (takeScreenShot) {
					takeScreenShot = false;
					canvas.toBlob(function(b) { saveAs(b, "screen.png"); }, "image/png");
				}

				// If we're dropping frames, decrease the sampling rate
				if (!newVolumeUpload && targetSamplingRate > samplingRate) {
					samplingRate = 0.8 * samplingRate + 0.2 * targetSamplingRate;
					gl.uniform1f(shader.uniforms["dt_scale"], samplingRate);
				}

				newVolumeUpload = false;
				startTime = endTime;
			}, targetFrameTime);
		} else {
			gl.deleteTexture(volumeTexture);
			volumeTexture = tex;
		}
	});
}

var selectColormap = function() {
	var selection = document.getElementById("colormapList").value;
	var colormapImage = new Image();
	colormapImage.onload = function() {
		gl.activeTexture(gl.TEXTURE1);
		gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 180, 1,
			gl.RGBA, gl.UNSIGNED_BYTE, colormapImage);
	};
	colormapImage.src = colormaps[selection];
}

window.onload = function(){
	fillcolormapSelector();
	updateProjectionMatrix(null);

	gl = canvas.getContext("webgl2");
	if (!gl) {
		alert("Unable to initialize WebGL2. Your browser may not support it");
		return;
	}

	// Setup accordions
	var acc = document.getElementsByClassName("accordion");
	var i;
	for (i = 0; i < acc.length; i++) {
	  acc[i].addEventListener("click", function() {
		/* Toggle between adding and removing the "active" class,
		to highlight the button that controls the panel */
		this.classList.toggle("active");
	
		/* Toggle between hiding and showing the active panel */
		var panel = this.nextElementSibling;
		if (panel.style.display === "block") {
		  panel.style.display = "none";
		} else {
		  panel.style.display = "block";
		}
	  });

	  // Hide the panel to start
	  var panel = acc[i].nextElementSibling;
	  panel.style.display = "none";
	} 

	// Register mouse and touch listeners
	var controller = new Controller();
	controller.mousemove = function(prev, cur, evt) {
		if (evt.buttons == 1) {
			camera.rotate(prev, cur);

		} else if (evt.buttons == 2) {
			camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
		}
	};
	controller.wheel = function(amt) { camera.zoom(amt); };
	controller.pinch = controller.wheel;
	controller.twoFingerDrag = function(drag) { camera.pan(drag); };

	document.addEventListener("keydown", function(evt) {
		if (evt.key == "p") {
			takeScreenShot = true;
		}
	});

	controller.registerForCanvas(canvas);

	// Setup VAO and VBO to render the cube to run the raymarching shader
	var vao = gl.createVertexArray();
	gl.bindVertexArray(vao);

	var vbo = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(cubeStrip), gl.STATIC_DRAW);

	gl.enableVertexAttribArray(0);
	gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

	shader = new Shader(gl, vertShader, fragShader);
	shader.use(gl);

	gl.uniform1i(shader.uniforms["volume"], 0);
	gl.uniform1i(shader.uniforms["colormap"], 1);
	gl.uniform1f(shader.uniforms["dt_scale"], 1.0);

	// Setup required OpenGL state for drawing the back faces and
	// composting with the background color
	gl.enable(gl.CULL_FACE);
	gl.cullFace(gl.FRONT);
	gl.enable(gl.BLEND);
	gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

	// Load the default colormap and upload it, after which we
	// load the default volume.
	var colormapImage = new Image();
	colormapImage.onload = function() {
		var colormap = gl.createTexture();
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, colormap);
		gl.texStorage2D(gl.TEXTURE_2D, 1, gl.SRGB8_ALPHA8, 180, 1);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);	
		gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 180, 1,
			gl.RGBA, gl.UNSIGNED_BYTE, colormapImage);

		selectVolume();
	};
	colormapImage.src = "colormaps/samsel-linear-green.png";
}

var updateProjectionMatrix = function(e) {
	var fov = document.getElementById("fovInput").value;
	// var ortho = document.getElementById("othoInput").checked;
	var ortho = false;

	canvas = document.getElementById("glcanvas");
	WIDTH = canvas.getBoundingClientRect()["width"];
	HEIGHT = canvas.getBoundingClientRect()["height"];

	if (camera != null) {
		camera = new ArcballCamera(camera.eyePos(), [-camera.centerTranslation[12], -camera.centerTranslation[13], -camera.centerTranslation[14]], camera.upDir(), zoom_increment, [WIDTH, HEIGHT]);
	}
	else {
		camera = new ArcballCamera(defaultEye, center, up, zoom_increment, [WIDTH, HEIGHT]);
	}

	projView = mat4.create();

	if (e != null && e.dataset.sync) {
		document.getElementById(e.dataset.sync).value = e.value;
	}

	if (ortho) {
		proj = mat4.ortho(mat4.create(), -0.5, 0.5, -0.5, WIDTH / HEIGHT, 0.1, 100);
	}
	else{
		proj = mat4.perspective(mat4.create(), fov * Math.PI / 180.0,
			WIDTH / HEIGHT, 0.1, 100);
	}
}

window.onresize = function(){
	updateProjectionMatrix(null);
}

var updateMiscValues = function(e) {	
	if (e.dataset.sync) {
		document.getElementById(e.dataset.sync).value = e.value;
	}

	near_clip = parseFloat(document.getElementById("nearClipInput").value);
	far_clip = parseFloat(document.getElementById("farClipInput").value);
	
	xMinLayerInput = parseFloat(document.getElementById("xMinLayerInput").value);
	xMaxLayerInput = parseFloat(document.getElementById("xMaxLayerInput").value);
	
	yMinLayerInput = parseFloat(document.getElementById("yMinLayerInput").value);
	yMaxLayerInput = parseFloat(document.getElementById("yMaxLayerInput").value);

	zMinLayerInput = parseFloat(document.getElementById("zMinLayerInput").value);
	zMaxLayerInput = parseFloat(document.getElementById("zMaxLayerInput").value);

	if (xMinLayerInput > xMaxLayerInput) {
		xMinLayerInput = xMaxLayerInput
	}

	if (yMinLayerInput > yMaxLayerInput) {
		yMinLayerInput = yMaxLayerInput
	}
	
	if (zMinLayerInput > zMaxLayerInput) {
		zMinLayerInput = zMaxLayerInput
	}

	isolateLayerInput = document.getElementById("isolateLayerInput").checked;

	if (isolateLayerInput) {
		bboxMin = [
			xMinLayerInput / volDims[0],
			yMinLayerInput / volDims[1],
			zMinLayerInput / volDims[2],
		]
		bboxMax = [
			xMaxLayerInput / volDims[0],
			yMaxLayerInput / volDims[1],
			zMinLayerInput / volDims[2] + 0.0001,
		]
	}
	else {
		bboxMin = [
			xMinLayerInput / volDims[0],
			yMinLayerInput / volDims[1],
			zMinLayerInput / volDims[2],
		]
		bboxMax = [
			xMaxLayerInput / volDims[0],
			yMaxLayerInput / volDims[1],
			zMaxLayerInput / volDims[2],
		]
	}
}

var fillcolormapSelector = function() {
	var selector = document.getElementById("colormapList");
	for (p in colormaps) {
		var opt = document.createElement("option");
		opt.value = p;
		opt.innerHTML = p;
		selector.appendChild(opt);
	}
}

function updateBackgroundColor(picker) {
	backgroundColor = [picker.channel('R') / 255, picker.channel('G') / 255, picker.channel('B') / 255]
}
