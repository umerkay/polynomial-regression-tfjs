var xs = [];
var ys = [];
var coeffs = [];
var minX = -1;
var maxX = 1;
var minY = 0;
var maxY = 1;

const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);
const polyDegree = parseInt(prompt("Enter the degree of the polynomial"));


function setup() {
  createCanvas(500, 500);
  coeffs = [];
  for (let i = 0; i <= polyDegree; i++) {
    coeffs.push(tf.variable(tf.scalar(random(-1, 1))));
  }
}

function draw() {
  tf.tidy(() => {
    if (xs.length > 0) {
      const tfyvals = tf.tensor1d(ys);
      optimizer.minimize(() => loss(f(xs), tfyvals));
    }
  });

  background(0);

  var lineXs = [];
  textSize(10);
  fill(255);
  strokeWeight(1);
  for (
    let i = minX;
    i <= maxX + (abs(maxX) + abs(minX)) / 40;
    i += (abs(maxX) + abs(minX)) / 40
  ) {
    lineXs.push(i);
  }
  const tflineYs = tf.tidy(() => f(lineXs));
  let lineYs = tflineYs.dataSync();
  tflineYs.dispose();
  lineYs = lineYs.map((y) => map(y, 0, 1, height, 0));
  lineXs = lineXs.map((x) => map(x, 0, 1, 0, width));

  beginShape();
  for (var i = lineXs.length - 1; i >= 0; i--) {
    vertex(lineXs[i], lineYs[i]);
  }
  stroke(255);
  strokeWeight(5);
  noFill();
  endShape();

  fill(255);
  stroke(0);
  strokeWeight(1);
  for (var i = 0; i < xs.length; i++) {
    ellipse(xs[i] * width, height - ys[i] * height, 10);
  }
}

function f(xvals) {
  const tfxvals = tf.tensor1d(xvals);
  let yvals = tf.zerosLike(tfxvals);

  for (let i = 0; i <= polyDegree; i++) {
    yvals = yvals.add(tfxvals.pow(tf.scalar(i)).mul(coeffs[i]));
  }

  return yvals;
}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  xs.push(x);
  ys.push(y);

  // const tfx = tf.scalar(x);
  // let xpow = tfx;
  // for (let i = 0; i <= polyDegree; i++) {
  //   coeffs[i].assign(coeffs[i].add(xpow.mul(y)));
  //   xpow = xpow.mul(tfx);
  // }
}

function mouseDragged() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  xs.push(x);
  ys.push(y);

  // const tfx = tf.scalar(x);
  // let xpow = tfx;
  // for (let i = 0; i <= polyDegree; i++) {
  //   coeffs[i].assign(coeffs[i].add(xpow.mul(y)));
  //   xpow = xpow.mul(tfx);
  // }
}
