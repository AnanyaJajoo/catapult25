const express = require('express');
const AWS = require('aws-sdk');
const cors = require('cors');
require('dotenv').config();

const app = express();

// Configure CORS
app.use(cors({
  origin: 'http://localhost:3000', // Frontend running on port 3000
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type'],
  credentials: true
}));

app.use(express.json());

// Log AWS configuration (without sensitive data)
console.log('AWS Configuration:', {
  region: process.env.AWS_REGION,
  bucket: process.env.AWS_BUCKET_NAME,
  hasAccessKey: !!process.env.AWS_ACCESS_KEY_ID,
  hasSecretKey: !!process.env.AWS_SECRET_ACCESS_KEY
});

// Configure AWS
const s3 = new AWS.S3({
  region: process.env.AWS_REGION || 'us-east-2',
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  signatureVersion: 'v4',
});

// Test AWS connection
async function testAWSConnection() {
  try {
    console.log('Testing AWS connection...');
    const params = {
      Bucket: process.env.AWS_BUCKET_NAME,
      MaxKeys: 1
    };
    const data = await s3.listObjectsV2(params).promise();
    console.log('AWS connection successful!');
    console.log('Bucket contents:', data.Contents ? data.Contents.length : 0, 'objects');
  } catch (error) {
    console.error('AWS connection failed:', error);
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    if (error.code === 'InvalidAccessKeyId') {
      console.error('The AWS Access Key ID is invalid');
    } else if (error.code === 'SignatureDoesNotMatch') {
      console.error('The AWS Secret Access Key is incorrect');
    } else if (error.code === 'NoSuchBucket') {
      console.error('The specified bucket does not exist');
    }
  }
}

// Test connection on startup
testAWSConnection();

// Health check endpoint
app.get('/', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' });
});

// Get presigned URL for upload
app.post('/get-presigned-url', async (req, res) => {
  try {
    console.log('Received request body:', req.body);
    const { fileName, fileType } = req.body;
    
    if (!fileName || !fileType) {
      console.error('Missing required fields:', { fileName, fileType });
      return res.status(400).json({ 
        error: 'Missing required fields',
        details: 'fileName and fileType are required' 
      });
    }

    console.log('Generating presigned URL for:', fileName);
    const params = {
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: `videos/${Date.now()}-${fileName}`,
      Expires: 60,
      ContentType: fileType,
    };

    console.log('S3 params:', params);
    const uploadURL = await s3.getSignedUrlPromise('putObject', params);
    console.log('Generated presigned URL successfully');
    
    const response = { url: uploadURL, key: params.Key };
    console.log('Sending response:', response);
    res.json(response);
  } catch (error) {
    console.error('Error generating presigned URL:', error);
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    res.status(500).json({ 
      error: 'Error generating upload URL',
      details: error.message 
    });
  }
});

// List videos from S3 bucket
app.get('/videos', async (req, res) => {
  try {
    console.log('Listing videos from bucket:', process.env.AWS_BUCKET_NAME);
    const params = {
      Bucket: process.env.AWS_BUCKET_NAME,
      Prefix: 'videos/'
    };

    const data = await s3.listObjectsV2(params).promise();
    console.log('Found', data.Contents ? data.Contents.length : 0, 'videos');
    
    const videos = data.Contents.map(item => {
      const key = item.Key.replace(/ /g, '+');
      const videoUrl = `https://${process.env.AWS_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${key}`;
      console.log('Generated video URL:', videoUrl);
      return {
        url: videoUrl,
        name: item.Key.split('/').pop(),
        key: item.Key
      };
    });

    console.log('Returning videos:', videos);
    res.json(videos);
  } catch (error) {
    console.error('Error listing videos:', error);
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    res.status(500).json({ 
      error: 'Error listing videos',
      details: error.message 
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 