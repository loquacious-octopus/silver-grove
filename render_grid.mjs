import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer';

const [, , modulePath, outputPath] = process.argv;
if (!modulePath || !outputPath) {
  console.error('usage: node render_grid.mjs module.mjs output.png');
  process.exit(2);
}

const threeModule = pathToFileURL(path.resolve('node_modules/three/build/three.module.js')).href;
const targetModule = pathToFileURL(path.resolve(modulePath)).href;
const html = `<!doctype html>
<html>
<body style="margin:0;background:#f4f4f4">
<canvas id="grid" width="1041" height="1041"></canvas>
<script type="module">
import * as THREE from ${JSON.stringify(threeModule)};
import generate from ${JSON.stringify(targetModule)};

const canvas = document.getElementById('grid');
const ctx = canvas.getContext('2d');
const views = [
  {pos: [1.9, 1.35, 2.1], label: 'front'},
  {pos: [-2.2, 1.25, 1.6], label: 'side'},
  {pos: [1.8, 2.2, -1.9], label: 'back'},
  {pos: [0.0, 3.1, 0.01], label: 'top'}
];
const size = 518;
const gap = 5;
const root = generate(THREE);
root.updateMatrixWorld(true);
const box = new THREE.Box3().setFromObject(root);
const center = new THREE.Vector3();
const dims = new THREE.Vector3();
box.getCenter(center);
box.getSize(dims);
const maxDim = Math.max(dims.x, dims.y, dims.z, 0.01);
root.position.sub(center);
root.scale.multiplyScalar(1.35 / maxDim);

async function drawView(view, index) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setSize(size, size);
  renderer.setPixelRatio(1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f4f4);
  const clone = root.clone(true);
  scene.add(clone);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x606060, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(3, 4, 5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.7);
  fill.position.set(-3, 2, -2);
  scene.add(fill);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  camera.position.set(...view.pos);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  const col = index % 2;
  const row = Math.floor(index / 2);
  ctx.drawImage(renderer.domElement, col * (size + gap), row * (size + gap));
  renderer.dispose();
}

try {
  for (let i = 0; i < views.length; i++) await drawView(views[i], i);
  window.__renderDone = true;
} catch (error) {
  window.__renderError = String(error && error.stack || error);
}
</script>
</body>
</html>`;

const htmlPath = path.join(path.dirname(path.resolve(outputPath)), 'render.html');
await fs.writeFile(htmlPath, html);

const browser = await puppeteer.launch({
  headless: 'new',
  args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--use-gl=swiftshader']
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1041, height: 1041, deviceScaleFactor: 1 });
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.waitForFunction('window.__renderDone || window.__renderError', { timeout: 30000 });
  const err = await page.evaluate('window.__renderError || ""');
  if (err) throw new Error(err);
  const canvas = await page.$('#grid');
  await canvas.screenshot({ path: outputPath });
} finally {
  await browser.close();
}
