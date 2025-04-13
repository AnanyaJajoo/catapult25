const express = require('express');
const cors = require('cors');
require('dotenv').config({ path: '../.env' });
const { spawn } = require('child_process');
const path = require('path');
const app = express();

// Configure CORS
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001'], // Allow both ports
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Endpoint to run the Python script
app.get('/run-python', (req, res) => {
  // Get the video filename from the query parameter
  const videoFilename = req.query.video;
  
  if (!videoFilename) {
    return res.status(400).json({ error: 'Video filename is required' });
  }
  
  console.log(`Running Python script with video: ${videoFilename}`);
  
  // Construct the full path to your Python script
  console.log(path.join(__dirname, 'pythonScript/webApp.py'));
  const scriptPath = path.join(__dirname, 'pythonScript/webApp.py');
  
  // Spawn the Python process
  const pythonProcess = spawn('/usr/bin/python3', [scriptPath, videoFilename]);

  let output = '';
  let errorOutput = '';

  pythonProcess.stdout.on('data', (data) => {
    const dataStr = data.toString();
    console.log(`Python stdout: ${dataStr}`);
    output += dataStr;
  });

  pythonProcess.stderr.on('data', (data) => {
    const dataStr = data.toString();
    console.error(`Python stderr: ${dataStr}`);
    errorOutput += dataStr;
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    if (code === 0) {
      res.json({ success: true, result: output });
    } else {
      res.status(500).json({ 
        success: false, 
        error: errorOutput || 'Python script failed without error output', 
        code 
      });
    }
  });
});

// Start the server
const PORT = process.env.BACKEND_PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});