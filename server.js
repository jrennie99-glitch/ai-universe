const express = require('express');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = 3500;

// Serve static files from spark dist
app.use('/spark', express.static(path.join(__dirname, '../spark/dist')));
app.use('/spark', express.static(path.join(__dirname, '../spark/examples')));

// Serve the main scene
app.use(express.static(path.join(__dirname, 'public')));

// API endpoints for sim engine integration
app.use(express.json());

// Current scene state
let sceneState = {
  environment: null,       // .ply file path
  characters: [],          // { id, name, position, animation, dialog }
  camera: { x: 0, y: 0, z: 3, target: 'none' },
  broadcast: { playing: false, fps: 30 }
};

app.get('/api/scene', (req, res) => {
  res.json(sceneState);
});

app.post('/api/scene', (req, res) => {
  Object.assign(sceneState, req.body);
  res.json({ ok: true, state: sceneState });
});

app.post('/api/character/add', (req, res) => {
  const char = req.body;
  sceneState.characters.push(char);
  res.json({ ok: true, character: char });
});

app.post('/api/camera/move', (req, res) => {
  sceneState.camera = { ...sceneState.camera, ...req.body };
  res.json({ ok: true, camera: sceneState.camera });
});

app.post('/api/broadcast/start', (req, res) => {
  sceneState.broadcast.playing = true;
  res.json({ ok: true });
});

app.post('/api/broadcast/stop', (req, res) => {
  sceneState.broadcast.playing = false;
  res.json({ ok: true });
});

app.listen(PORT, () => {
  console.log(`🔥 AI World 3D running at http://localhost:${PORT}`);
  console.log(`   On your network: http://YOUR_IP:${PORT}`);
});
