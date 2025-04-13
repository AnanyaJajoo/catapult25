import AWS from 'aws-sdk';

// Log AWS configuration (without sensitive data)
console.log('AWS Configuration:', {
  region: process.env.REACT_APP_REGION,
  bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
  hasIdentityPoolId: !!process.env.REACT_APP_IDENTITY_POOL_ID,
  identityPoolId: process.env.REACT_APP_IDENTITY_POOL_ID
});

// Configure AWS SDK with direct credentials
AWS.config.update({
  region: process.env.REACT_APP_REGION,
  accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY
});

console.log('AWS SDK Config:', {
  region: process.env.REACT_APP_REGION,
  hasCredentials: !!process.env.REACT_APP_IDENTITY_POOL_ID
});

// Create S3 instance
const s3 = new AWS.S3({
  apiVersion: '2006-03-01',
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
    console.log('S3 params:', params);
    const data = await s3.listObjectsV2(params).promise();
    console.log('S3 connection successful!');
    console.log('Bucket contents:', data.Contents ? data.Contents.length : 0, 'objects');
  } catch (error) {
    console.error('S3 connection failed:', error);
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

// Run the test
testS3Connection();

export { s3 }; 