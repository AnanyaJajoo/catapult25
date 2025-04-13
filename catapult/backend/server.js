const express = require('express');
const cors = require('cors');
const fs = require('fs/promises'); // <-- for readFile
const path = require('path');
const { spawn } = require('child_process');
require('dotenv').config({ path: '../.env' });

const app = express();

// Configure CORS
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001'],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Run Python script
app.get('/run-python', async (req, res) => {
  const videoFilename = req.query.video;
  const videoKey = req.query.key;

  if (!videoFilename || !videoKey) {
    return res.status(400).json({ error: 'Video filename and key are required' });
  }

  const scriptPath = path.join(__dirname, 'pythonScript/webApp.py');
  const pythonProcess = spawn('/usr/bin/python3', [scriptPath, videoFilename, videoKey]);

  let output = '';
  let errorOutput = '';

  pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  pythonProcess.on('close', (code) => {
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

// Run Python script for alert email
app.get('/alert-email', async (req, res) => {
  const videoFilename = req.query.video;

  if (!videoFilename) {
    return res.status(400).json({ error: 'Video filename is required' });
  }

  // Path to the Python script
  const scriptPath = path.join(__dirname, 'pythonScript/autoEmail.py');
  
  // Pass videoFilename as an argument to the Python script
  const pythonProcess = spawn('/usr/bin/python3', [scriptPath, videoFilename]);

  let output = '';
  let errorOutput = '';

  pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  pythonProcess.on('close', (code) => {
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

function parseEventTextFile(content) {
  const lines = content.split('\n').map(line => line.trim()).filter(line => line.length > 0);
  const result = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Match labeled events like "2.0 - No, Yes, No."
    const labeledMatch = line.match(/^([\d.]+)\s*-\s*([^,]+),\s*([^,]+),\s*([^,]+)$/);
    if (labeledMatch && i + 1 < lines.length && !lines[i + 1].includes(' - ')) {
      const time = parseFloat(labeledMatch[1]);
      const description = lines[i + 1];

      // Determine event type based on yes/no chain
      let eventType = '';
      const firstYes = labeledMatch[2].toLowerCase().trim();
      const secondYes = labeledMatch[3].toLowerCase().trim();
      const thirdYes = labeledMatch[4].toLowerCase().trim();

      if (firstYes === 'yes') eventType = 'fallen';
      if (secondYes === 'yes') eventType = 'dangerous';
      if (thirdYes === 'yes') eventType = 'anger';

      result.push({ time, description, eventType });
      i++; // Skip the next line (already processed)
      continue;
    }

    // Match audio-only events (e.g., "0.542 - loud")
    const audioMatch = line.match(/^([\d.]+)\s*-\s*loud$/i);
    if (audioMatch) {
      const time = parseFloat(audioMatch[1]);
      result.push({ time, description: 'loud', eventType: 'loud' });
    }

    // Match help events (e.g., "2.0 - help")
    const helpMatch = line.match(/^([\d.]+)\s*-\s*help$/i);
    if (helpMatch) {
      const time = parseFloat(helpMatch[1]);
      result.push({ time, description: 'help', eventType: 'help' });
    }
  }

  // Sort events by time in ascending order (earliest first)
  result.sort((a, b) => a.time - b.time);

  return result;
}

// Get and parse a text file from /output
app.get('/get-text-file', async (req, res) => {
  const fileKey = req.query.fileKey;

  if (!fileKey) {
    return res.status(400).json({ error: 'fileKey query parameter is required' });
  }

  try {
    const trimmedKey = fileKey.substring(7, fileKey.length - 4);
    const filePath = path.join(__dirname, 'output', `${trimmedKey}.txt`);
    const content = await fs.readFile(filePath, 'utf-8');
    const parsed = parseEventTextFile(content);
    res.json(parsed);
  } catch (err) {
    console.error('Failed to read or parse text file:', err);
    res.status(404).json({ error: 'File not found or unreadable' });
  }
});

// Start server
const PORT = process.env.BACKEND_PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
