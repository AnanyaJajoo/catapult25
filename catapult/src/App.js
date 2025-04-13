import { useState, useEffect, useRef } from 'react';
import { Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { useAuth } from 'react-oidc-context';
import AWS from 'aws-sdk';
import './App.css';

function NavBar({ onSignOut }) {
  const location = useLocation();


  return (
    <nav className="nav-bar">
      <div className="nav-buttons">
        <Link to="/" className={`nav-button ${location.pathname === '/' ? 'active' : ''}`}>
          All Videos
        </Link>
        <Link to="/favorites" className={`nav-button ${location.pathname === '/favorites' ? 'active' : ''}`}>
          Favorites
        </Link>
        <Link to="/realtime" className={`nav-button ${location.pathname === '/realtime' ? 'active' : ''}`}>
          Realtime
        </Link>
        <button className="signout-button" onClick={onSignOut}>Sign Out</button>
      </div>
    </nav>
  );
}

function App() {
  const auth = useAuth();
  const [videos, setVideos] = useState([]);
  const fileInputRef = useRef(null);
  const [s3Client, setS3Client] = useState(null);
  const [authError, setAuthError] = useState('');
  
  // Add this effect to detect return from authentication
  useEffect(() => {
    // Check if we're returning from auth redirect (has 'code' in URL)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('code')) {
      console.log("Auth code detected in URL - likely returning from authentication");
      
      // Clean up the URL by removing the code
      window.history.replaceState({}, document.title, window.location.pathname);
      
      // The auth library should handle the code automatically
      // We just need to wait for auth state to update
    }
  }, []);
  
  // Debug authentication state with more detailed logging
  useEffect(() => {
    console.log('Auth state updated:', {
      isLoading: auth.isLoading,
      isAuthenticated: auth.isAuthenticated,
      activeNavigator: auth.activeNavigator,
      hasUser: !!auth.user,
      error: auth.error
    });
    
    if (auth.isAuthenticated && auth.user) {
      console.log("User authenticated with claims:", auth.user.profile);
    }
    
    if (auth.error) {
      console.error('Auth error details:', auth.error);
      setAuthError(auth.error.message);
    }
  }, [auth.isLoading, auth.isAuthenticated, auth.activeNavigator, auth.user, auth.error]);
  
  // Configure S3 client when authentication token is available
  useEffect(() => {
    if (auth.isAuthenticated && auth.user) {
      console.log("User authenticated, configuring S3...");
      
      // Try up to 3 times with increasing delays
      const tryConfigureS3 = (attempt = 1) => {
        configureS3Client().catch(err => {
          console.error(`S3 configuration error (attempt ${attempt}):`, err);
          
          if (attempt < 3) {
            console.log(`Retrying in ${attempt * 2} seconds...`);
            setTimeout(() => tryConfigureS3(attempt + 1), attempt * 2000);
          }
        });
      };
      
      tryConfigureS3();
    }
  }, [auth.isAuthenticated, auth.user]);

  
  // Sign out handler with improved error handling
  const handleSignOut = () => {
    try {
      auth.removeUser();
      const clientId = process.env.REACT_APP_USER_POOL_WEB_CLIENT_ID;
      const logoutUri = encodeURIComponent(process.env.REACT_APP_REDIRECT_URI);
      const cognitoDomain = process.env.REACT_APP_COGNITO_DOMAIN;
      
      if (!cognitoDomain) {
        console.error("Cognito domain is not defined");
        return;
      }
      
      // Handle trailing slash in the domain
      const domainWithoutTrailingSlash = cognitoDomain.endsWith('/') 
        ? cognitoDomain.slice(0, -1) 
        : cognitoDomain;
        
      window.location.href = `${domainWithoutTrailingSlash}/logout?client_id=${clientId}&logout_uri=${logoutUri}`;
    } catch (error) {
      console.error("Error during sign out:", error);
      // Force clear local storage as fallback
      localStorage.clear();
      sessionStorage.clear();
      window.location.href = process.env.REACT_APP_REDIRECT_URI;
    }
  };
  
  // Handle sign-in click with error handling
  const handleSignIn = () => {
    try {
      // Clear any previous auth errors
      setAuthError('');
      
      // Hard-code the URL completely for testing
      const loginUrl = 'https://us-east-2dzzxd9fgw.auth.us-east-2.amazoncognito.com/login?client_id=qao1r6646c227dp1hspr4ptrj&response_type=code&scope=email+openid+phone&redirect_uri=http%3A%2F%2Flocalhost%3A3000';
      
      console.log("Redirecting to hard-coded URL:", loginUrl);
      
      // Redirect to the login URL
      window.location.href = loginUrl;
    } catch (error) {
      console.error("Exception during sign-in:", error);
      setAuthError(`Sign-in exception: ${error.message}`);
    }
  };
  
  // Configure S3 client when authentication token is available
  useEffect(() => {
    if (auth.isAuthenticated && auth.user) {
      console.log("User authenticated, configuring S3...");
      configureS3Client().catch(err => {
        console.error("S3 configuration error:", err);
      });
    }
  }, [auth.isAuthenticated, auth.user]);

  // Configure S3 client with improved error handling
  const configureS3Client = async () => {
    try {
      console.log("Starting S3 client configuration");
      
      // Set up AWS credentials using the OIDC tokens
      AWS.config.region = process.env.REACT_APP_REGION || 'us-east-2';
      
      // Debug token
      console.log("ID Token available:", !!auth.user?.id_token);
      
      // Create credentials using Cognito Identity Pool
      AWS.config.credentials = new AWS.CognitoIdentityCredentials({
        IdentityPoolId: process.env.REACT_APP_IDENTITY_POOL_ID,
        Logins: {
          [`cognito-idp.${process.env.REACT_APP_REGION}.amazonaws.com/${process.env.REACT_APP_USER_POOL_ID}`]: auth.user.id_token
        }
      });
      
      console.log("Refreshing AWS credentials...");
      
      // Refresh the credentials
      await new Promise((resolve, reject) => {
        AWS.config.credentials.refresh(err => {
          if (err) {
            console.error("Credential refresh error:", err);
            reject(err);
          } else {
            console.log("AWS credentials refreshed successfully");
            resolve();
          }
        });
      });
      
      console.log("Creating S3 client...");
      
      // Create a new S3 instance
      const s3 = new AWS.S3({
        region: process.env.REACT_APP_REGION || 'us-east-2',
        params: { Bucket: process.env.REACT_APP_AWS_BUCKET_NAME || 'catapult2025' }
      });
      
      setS3Client(s3);
      
      // Get the username from the token claims
      const username = auth.user.profile.email || auth.user.profile.sub;
      console.log("User identified as:", username);
      
      // Create folder for user and fetch videos
      await createUserFolder(username, s3);
      await fetchVideos(username, s3);
      
      return s3;
    } catch (error) {
      console.error('Error configuring S3:', error);
      return null;
    }
  };

  // Function to create a folder for the user
  const createUserFolder = async (username, s3) => {
    try {
      // In S3, folders are just objects with "/" at the end
      const params = {
        Bucket: process.env.AWS_BUCKET_NAME || 'catapult2025',
        Key: `${username}/`,
        Body: ''
      };
      await s3.putObject(params).promise();
      console.log(`Folder created for user: ${username}`);
    } catch (error) {
      console.error('Error creating folder:', error);
    }
  };

  // Fetch videos from user's folder in S3
  const fetchVideos = async (username, s3) => {
    try {
      const params = {
        Bucket: process.env.AWS_BUCKET_NAME || 'catapult2025',
        Prefix: `${username}/`,
      };

      const data = await s3.listObjectsV2(params).promise();
      
      if (data.Contents) {
        const videoList = await Promise.all(
          data.Contents
            .filter(item => !item.Key.endsWith('/')) // Filter out folder objects
            .map(async (item) => {
              // Generate a signed URL for each video
              const urlParams = {
                Bucket: process.env.AWS_BUCKET_NAME || 'catapult2025',
                Key: item.Key,
                Expires: 3600 // URL expires in 1 hour
              };
              
              const url = s3.getSignedUrl('getObject', urlParams);
              return {
                url,
                name: item.Key.split('/').pop(),
                isFavorite: false
              };
            })
        );
        setVideos(videoList);
      }
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  // Handle file upload
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file && s3Client && auth.user) {
      try {
        const username = auth.user.profile.email || auth.user.profile.sub;
        const fileName = `${Date.now()}-${file.name}`;
        
        const uploadParams = {
          Bucket: process.env.AWS_BUCKET_NAME || 'catapult2025',
          Key: `${username}/${fileName}`,
          Body: file,
          ContentType: file.type
        };
        
        await s3Client.upload(uploadParams).promise();
        console.log('Upload successful');
        
        // Refresh the video list
        fetchVideos(username, s3Client);
      } catch (error) {
        console.error('Error uploading file:', error);
      }
    }
  };

  // Handle upload button click
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };
  
  // Toggle favorite status
  const toggleFavorite = (index) => {
    const updatedVideos = [...videos];
    updatedVideos[index].isFavorite = !updatedVideos[index].isFavorite;
    setVideos(updatedVideos);
  };
  
  // Handle loading and error states
  if (auth.isLoading) {
    return <div className="loading">Loading...</div>;
  }

  // Handle active navigator state (happens during redirect)
  if (auth.activeNavigator) {
    return <div className="loading">Redirecting for authentication...</div>;
  }

  if (auth.error) {
    return <div className="error">Encountering error: {auth.error.message}</div>;
  }

  // Handle error state with more details
  if (auth.error) {
    return (
      <div className="error-container">
        <h2>Authentication Error</h2>
        <p>{auth.error.message}</p>
        <div className="error-details">
          <pre>{JSON.stringify(auth.error, null, 2)}</pre>
        </div>
        <button onClick={() => window.location.href = process.env.REACT_APP_REDIRECT_URI}>
          Return to Home
        </button>
      </div>
    );
  }

  // Handle unauthenticated state
  if (!auth.isAuthenticated) {
    return (
      <div className="login-container">
        <h1>Catapult Video Manager</h1>
        <button className="login-button" onClick={handleSignIn}>
          Sign in with Cognito
        </button>
      </div>
    );
  }

  // Render authenticated app
  return (
    <div className="App">
      <NavBar onSignOut={handleSignOut} />
      <div className="user-info">
        <p>Signed in as: {auth.user?.profile.email || auth.user?.profile.sub}</p>
      </div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        style={{ display: 'none' }}
      />
      <button className="upload-button" onClick={handleUploadClick}>
        Upload Video
      </button>
      
      <Routes>
        <Route path="/" element={
          <VideoGrid 
            videos={videos} 
            onToggleFavorite={toggleFavorite}
          />
        } />
        <Route path="/favorites" element={
          <VideoGrid 
            videos={videos.filter(v => v.isFavorite)} 
            onToggleFavorite={toggleFavorite}
          />
        } />
        <Route path="/realtime" element={
          <div className="realtime-placeholder">Realtime feature coming soon</div>
        } />
      </Routes>
    </div>
  );
}

function VideoGrid({ videos, onToggleFavorite }) {
  return (
    <div className="video-grid">
      {videos.length === 0 ? (
        <h3>No videos found</h3>
      ) : (
        videos.map((video, index) => (
          <div key={index} className="video-thumbnail">
            <video
              src={video.url}
              className="thumbnail-preview"
              controls
            />
            <div className="video-info">
              <span className="video-name">{video.name}</span>
              <button 
                className={`favorite-button ${video.isFavorite ? 'favorited' : ''}`}
                onClick={() => onToggleFavorite(index)}
              >
                ★
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

export default App;