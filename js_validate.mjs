import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as THREE from 'three';

const [, , modulePath] = process.argv;

function fail(rule, detail) {
  console.log(JSON.stringify({
    passed: false,
    stagesRun: ['module_load'],
    failures: [{ rule, detail }],
    metrics: {}
  }));
  process.exit(0);
}

if (!modulePath) fail('NO_MODULE', 'missing module path');

const started = performance.now();
let mod;
try {
  mod = await import(pathToFileURL(path.resolve(modulePath)).href + `?t=${Date.now()}`);
} catch (error) {
  fail('MODULE_IMPORT_FAILED', String(error && error.stack || error));
}
const moduleLoadMs = performance.now() - started;

if (typeof mod.default !== 'function') {
  fail('MISSING_DEFAULT_EXPORT', 'module must default-export function generate(THREE)');
}

let root;
const execStarted = performance.now();
try {
  root = mod.default(THREE);
} catch (error) {
  fail('EXECUTION_THREW', String(error && error.stack || error).slice(0, 1000));
}
const executionMs = performance.now() - execStarted;

if (!root || !root.isObject3D) {
  fail('INVALID_RETURN', 'generate(THREE) must return a THREE.Object3D');
}

let objectCount = 0;
let meshCount = 0;
let vertices = 0;
let materialCount = 0;
let textureBytes = 0;
let maxDepth = 0;

root.updateMatrixWorld(true);
root.traverse((obj) => {
  objectCount += 1;
  let depth = 0;
  let cursor = obj.parent;
  while (cursor) {
    depth += 1;
    cursor = cursor.parent;
  }
  maxDepth = Math.max(maxDepth, depth);
  if (obj.isMesh) {
    meshCount += 1;
    const pos = obj.geometry && obj.geometry.getAttribute && obj.geometry.getAttribute('position');
    if (pos && Number.isFinite(pos.count)) vertices += pos.count;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    materialCount += mats.filter(Boolean).length;
  }
});

const box = new THREE.Box3().setFromObject(root);
const failures = [];
if (box.isEmpty()) {
  failures.push({ rule: 'EMPTY_BOUNDS', detail: 'object bounds are empty' });
} else {
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  for (const axis of ['x', 'y', 'z']) {
    if (!Number.isFinite(size[axis]) || !Number.isFinite(center[axis])) {
      failures.push({ rule: 'NONFINITE_BOUNDS', detail: `${axis} bounds are not finite` });
    }
  }
  const maxDim = Math.max(size.x, size.y, size.z);
  const minDim = Math.min(size.x, size.y, size.z);
  if (maxDim <= 0.01) failures.push({ rule: 'TOO_SMALL', detail: `max dimension ${maxDim}` });
  if (maxDim > 2.5 || Math.max(Math.abs(center.x), Math.abs(center.y), Math.abs(center.z)) > 1.5) {
    failures.push({ rule: 'BOUNDING_BOX_OUT_OF_RANGE', detail: `center=${center.toArray()} size=${size.toArray()}` });
  }
  if (minDim <= 0 && meshCount > 0) {
    failures.push({ rule: 'DEGENERATE_BOUNDS', detail: `size=${size.toArray()}` });
  }
}

if (meshCount < 1) failures.push({ rule: 'NO_MESHES', detail: 'object contains no meshes' });
if (vertices > 250000) failures.push({ rule: 'TOO_MANY_VERTICES', detail: `${vertices} vertices` });

console.log(JSON.stringify({
  passed: failures.length === 0,
  stagesRun: ['module_load', 'execute', 'bounds', 'complexity'],
  failures,
  moduleLoadMs,
  executionMs,
  totalMs: performance.now() - started,
  metrics: {
    objects: objectCount,
    meshes: meshCount,
    vertices,
    drawCalls: meshCount,
    maxDepth,
    materials: materialCount,
    textureBytes,
    bbox: box.isEmpty() ? null : {
      min: { x: box.min.x, y: box.min.y, z: box.min.z },
      max: { x: box.max.x, y: box.max.y, z: box.max.z }
    }
  }
}));
