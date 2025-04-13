import AWS from 'aws-sdk';

// Configure AWS
AWS.config.update({
  region: process.env.REACT_APP_REGION,
  accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY
});

// Create S3 instance
const s3 = new AWS.S3({
  params: {
    Bucket: process.env.REACT_APP_AWS_BUCKET_NAME
  }
});

// Test S3 connection
async function testS3Connection() {
  try {
    console.log('Testing S3 connection...');
    console.log('Using bucket:', process.env.REACT_APP_AWS_BUCKET_NAME);
    const params = {
      Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
      MaxKeys: 1
    };
    const data = await s3.listObjectsV2(params).promise();
    console.log('S3 connection successful!');
    console.log('Bucket contents:', data.Contents ? data.Contents.length : 0, 'objects');
  } catch (error) {
    console.error('S3 connection failed:', error);
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
  }
}

// Run the test
testS3Connection();

export { s3 }; 