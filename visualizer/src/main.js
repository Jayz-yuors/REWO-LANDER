import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import './style.css';

const sceneRoot = document.querySelector('#scene');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#080f1d');
scene.fog = new THREE.Fog('#080f1d', 28, 100);

const camera = new THREE.PerspectiveCamera(43, 1, 0.1, 250);
camera.position.set(12, 11, 17);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
sceneRoot.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 3, 0);
controls.enableDamping = true;
controls.maxDistance = 38;
controls.minDistance = 5;

scene.add(new THREE.HemisphereLight(0x9cb9e8, 0x1b1c22, 2.0));
const moonLight = new THREE.DirectionalLight(0xffe4bd, 3.0);
moonLight.position.set(-8, 15, 9);
scene.add(moonLight);
const cyanLight = new THREE.PointLight(0x28d9ff, 55, 22);
cyanLight.position.set(0, 1.5, 0);
scene.add(cyanLight);

const moonMaterial = new THREE.MeshStandardMaterial({ color: 0x353b46, roughness: 0.97, metalness: 0.02 });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(160, 160, 1, 1), moonMaterial);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.08;
scene.add(ground);

// Landing pad and rings.
const pad = new THREE.Mesh(
  new THREE.CylinderGeometry(2.4, 2.4, 0.08, 80),
  new THREE.MeshStandardMaterial({ color: 0x182938, metalness: 0.6, roughness: 0.45, emissive: 0x06272e, emissiveIntensity: 0.8 })
);
pad.position.set(0, 0.02, 0);
scene.add(pad);
for (const [radius, color, opacity] of [[2.15, 0x37e8df, 0.9], [2.35, 0x73a7b8, 0.45], [1.25, 0x73a7b8, 0.35]]) {
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(radius, 0.025, 8, 100),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity })
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.09;
  scene.add(ring);
}
const crossMaterial = new THREE.MeshBasicMaterial({ color: 0xa5f9ee, transparent: true, opacity: 0.7 });
for (const [x, z, sx, sz] of [[0,0,1.5,0.035],[0,0,0.035,1.5]]) {
  const line = new THREE.Mesh(new THREE.BoxGeometry(sx, 0.025, sz), crossMaterial);
  line.position.set(x, 0.10, z);
  scene.add(line);
}

// Craters as dark, low circular depressions / rims.
for (let i = 0; i < 34; i++) {
  const angle = i * 2.399963;
  const radius = 5 + (i % 7) * 2.3;
  const r = 0.22 + (i % 5) * 0.16;
  const crater = new THREE.Mesh(
    new THREE.RingGeometry(r * 0.62, r, 24),
    new THREE.MeshStandardMaterial({ color: i % 2 ? 0x242b36 : 0x59606a, side: THREE.DoubleSide, roughness: 1 })
  );
  crater.rotation.x = -Math.PI / 2;
  crater.position.set(Math.cos(angle) * radius, 0.015, Math.sin(angle) * radius);
  scene.add(crater);
}

// Star field.
const starPositions = [];
for (let i = 0; i < 1200; i++) {
  starPositions.push((Math.random() - 0.5) * 180, 8 + Math.random() * 90, (Math.random() - 0.5) * 180);
}
const starGeometry = new THREE.BufferGeometry();
starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
scene.add(new THREE.Points(starGeometry, new THREE.PointsMaterial({ color: 0xd3e7ff, size: 0.11, sizeAttenuation: true })));

const landerGroup = new THREE.Group();
scene.add(landerGroup);
let modelRoot = null;
let modelLoaded = false;
let modelHeight = 2.5;
const loader = new GLTFLoader();
loader.load('/models/apollo-lunar-module.glb', (gltf) => {
  modelRoot = gltf.scene;
  // Normalize the downloaded GLB robustly. Update world matrices before
  // measuring because GLB nodes can carry nested transforms.
  modelRoot.updateMatrixWorld(true);
  let box = new THREE.Box3().setFromObject(modelRoot);
  const sizeBefore = new THREE.Vector3();
  box.getSize(sizeBefore);
  const maxDim = Math.max(sizeBefore.x, sizeBefore.y, sizeBefore.z);

  if (!Number.isFinite(maxDim) || maxDim < 1e-5) {
    throw new Error('GLB has no measurable geometry');
  }

  const scale = 2.4 / maxDim;
  modelRoot.scale.setScalar(scale);
  modelRoot.updateMatrixWorld(true);
  box = new THREE.Box3().setFromObject(modelRoot);
  const center = new THREE.Vector3();
  const scaledSize = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(scaledSize);

  // Center the actual transformed geometry around the visual pivot.
  modelRoot.position.x -= center.x;
  modelRoot.position.y -= center.y;
  modelRoot.position.z -= center.z;
  modelRoot.updateMatrixWorld(true);

  // Keep the model upright; adjust only if the asset proves to be oriented
  // differently when inspected in the viewport.
  landerGroup.add(modelRoot);
  modelHeight = Math.max(scaledSize.y, 1.4);
  modelLoaded = true;
  document.querySelector('#sceneStatus').textContent =
    `GLB LOADED · ${scaledSize.x.toFixed(1)} × ${scaledSize.y.toFixed(1)} × ${scaledSize.z.toFixed(1)}`;
  console.info('Apollo GLB loaded', { scale, size: scaledSize, center });
}, undefined, (err) => {
  console.error('GLB model failed to load:', err);
  document.querySelector('#sceneStatus').textContent = 'MODEL LOAD ERROR — CHECK FILE PATH';

  // Visible diagnostic fallback: confirms the scene/camera works even if
  // the GLB fails to load. It is not a replacement for the Apollo model.
  const fallback = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.42, 0.58, 1.15, 12),
    new THREE.MeshStandardMaterial({ color: 0xd8d5c8, metalness: 0.25, roughness: 0.6 })
  );
  body.position.y = 0.25;
  fallback.add(body);
  const top = new THREE.Mesh(
    new THREE.ConeGeometry(0.42, 0.55, 12),
    new THREE.MeshStandardMaterial({ color: 0xb8c4ce, metalness: 0.25, roughness: 0.5 })
  );
  top.position.y = 1.1;
  fallback.add(top);
  const legMaterial = new THREE.MeshStandardMaterial({ color: 0xe0b35b, metalness: 0.45, roughness: 0.5 });
  for (const sign of [-1, 1]) {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.0, 6), legMaterial);
    leg.position.set(sign * 0.62, -0.35, 0);
    leg.rotation.z = sign * -0.48;
    fallback.add(leg);
  }
  landerGroup.add(fallback);
});

// Engine plumes are visual effects, not part of the physics model.
function makePlume(color, length, radius) {
  const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, depthWrite: false });
  const mesh = new THREE.Mesh(new THREE.ConeGeometry(radius, length, 16, 1, true), material);
  mesh.rotation.x = Math.PI; // point down by default
  mesh.visible = false;
  landerGroup.add(mesh);
  return mesh;
}
const mainPlume = makePlume(0x42eaff, 1.8, 0.36);
const leftPlume = makePlume(0xffb44f, 0.65, 0.15);
const rightPlume = makePlume(0xffb44f, 0.65, 0.15);
mainPlume.position.set(0, -1.35, 0);
leftPlume.position.set(-0.7, -0.45, 0);
rightPlume.position.set(0.7, -0.45, 0);
const plumeGlow = new THREE.PointLight(0x31dfff, 0, 8);
landerGroup.add(plumeGlow);
plumeGlow.position.set(0, -1.1, 0);

const trajectoryLineMaterial = new THREE.LineBasicMaterial({ color: 0x69eee4, transparent: true, opacity: 0.7 });
let trajectoryLine = null;
let frames = [];
let episode = null;
let frameIndex = 0;
let playing = false;
let speed = 1;
let lastTime = 0;
let fractionalFrame = 0;

const byId = (id) => document.getElementById(id);
const actionNames = ['NO THRUST', 'LEFT ENGINE', 'MAIN ENGINE', 'RIGHT ENGINE'];
const setText = (id, value) => { byId(id).textContent = value; };

function stateToPosition(obs) {
  // LunarLander observation is normalized 2D state. These are presentation
  // scales only; the displayed scene is not a physical 3D solution.
  return new THREE.Vector3(obs[0] * 4.2, Math.max(0.65, 2.2 + obs[1] * 3.0), 0);
}
function updateTelemetry(frame) {
  const o = frame.observation;
  setText('stepReadout', String(frame.step).padStart(4, '0') + ' ');
  setText('returnReadout', Number(frame.total_reward).toFixed(2));
  setText('xVal', Number(o[0]).toFixed(3));
  setText('yVal', Number(o[1]).toFixed(3));
  setText('vxVal', Number(o[2]).toFixed(3));
  setText('vyVal', Number(o[3]).toFixed(3));
  setText('angleVal', Number(o[4]).toFixed(3));
  setText('angVelVal', Number(o[5]).toFixed(3));
  setText('leftContact', o[6] > 0.5 ? 'CONTACT' : 'NO');
  setText('rightContact', o[7] > 0.5 ? 'CONTACT' : 'NO');
  setText('actionNumber', `ACTION ${frame.action}`);
  setText('actionName', actionNames[frame.action] || 'UNKNOWN');
  setText('rewardReadout', Number(frame.reward).toFixed(3));
  byId('leftThruster').classList.toggle('active', frame.action === 1);
  byId('mainThruster').classList.toggle('active', frame.action === 2);
  byId('rightThruster').classList.toggle('active', frame.action === 3);
  const maxStep = Math.max(1, frames.length - 1);
  byId('progressBar').style.width = `${100 * frame.step / maxStep}%`;
}
function renderFrame(index) {
  if (!frames.length) return;
  frameIndex = Math.max(0, Math.min(frames.length - 1, index));
  const frame = frames[Math.floor(frameIndex)];
  const o = frame.observation;
  const target = stateToPosition(o);
  landerGroup.position.copy(target);
  // Gym angle is radians; invert sign for the viewer's coordinate convention.
  landerGroup.rotation.z = -o[4];
  // Engine effects follow discrete action indices.
  mainPlume.visible = frame.action === 2;
  leftPlume.visible = frame.action === 1;
  rightPlume.visible = frame.action === 3;
  plumeGlow.intensity = frame.action === 2 ? 45 : 0;
  const pulse = 0.85 + 0.15 * Math.sin(performance.now() * 0.025);
  mainPlume.scale.set(1, pulse, 1);
  if (trajectoryLine) scene.remove(trajectoryLine);
  const points = frames.slice(0, Math.floor(frameIndex) + 1).map(f => stateToPosition(f.observation).add(new THREE.Vector3(0, 0.1, 0.15)));
  if (points.length > 1) {
    trajectoryLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), trajectoryLineMaterial);
    scene.add(trajectoryLine);
  }
  updateTelemetry(frame);
}
async function loadTrajectory() {
  try {
    const response = await fetch(`/data/trajectory.json?t=${Date.now()}`);
    if (!response.ok) throw new Error('trajectory.json not found');
    const data = await response.json();
    episode = data.episodes?.[0];
    if (!episode?.frames?.length) throw new Error('No episode frames in trajectory.json');
    frames = episode.frames;
    setText('policyReadout', episode.policy.toUpperCase());
    setText('episodeStatus', `${data.environment} · ${episode.steps} recorded steps`);
    renderFrame(0);
  } catch (error) {
    console.error(error);
    setText('episodeStatus', 'No trajectory found. Run scripts/export_trajectory.py first.');
    setText('sceneStatus', 'WAITING FOR TRAJECTORY');
  }
}

byId('playBtn').addEventListener('click', () => {
  if (!frames.length) return;
  if (frameIndex >= frames.length - 1) renderFrame(0);
  playing = !playing;
  setText('playBtn', playing ? 'Ⅱ PAUSE' : '▶ PLAY');
});
byId('restartBtn').addEventListener('click', () => {
  playing = false;
  setText('playBtn', '▶ PLAY');
  fractionalFrame = 0;
  renderFrame(0);
});
byId('speedRange').addEventListener('input', (event) => {
  speed = Number(event.target.value);
  setText('speedLabel', `${speed.toFixed(2).replace(/0$/, '')}×`);
});

function resize() {
  const width = sceneRoot.clientWidth;
  const height = sceneRoot.clientHeight;
  if (!width || !height) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}
window.addEventListener('resize', resize);
resize();
loadTrajectory();

function animate(time) {
  requestAnimationFrame(animate);
  const delta = lastTime ? Math.min((time - lastTime) / 1000, 0.08) : 0;
  lastTime = time;
  if (playing && frames.length > 1) {
    fractionalFrame += delta * 30 * speed;
    if (fractionalFrame >= frames.length - 1) {
      fractionalFrame = frames.length - 1;
      playing = false;
      setText('playBtn', '▶ PLAY');
      const last = frames[frames.length - 1];
      setText('episodeStatus', last.terminated ? 'Episode ended (terminated).' : last.truncated ? 'Episode time limit reached.' : 'Replay finished.');
    }
    renderFrame(Math.floor(fractionalFrame));
  }
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);
