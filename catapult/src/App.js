import { useState, useRef, useEffect } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { s3 } from './aws-config';
import './App.css';

function NavBar() {
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
        <div className="search-container">
          <input type="text" placeholder="Search..." className="search-input" />
        </div>
      </div>
    </nav>
  );
}

function VideoGrid({ videos, onVideoClick, onToggleFavorite, showFavoritesOnly = false }) {
  const displayedVideos = showFavoritesOnly 
    ? videos.filter(video => video.isFavorite)
    : videos;

  return (
    <div className="video-grid">
      {displayedVideos.map((video, index) => {
        console.log('Rendering video:', video);
        return (
          <div 
            key={index} 
            className="video-thumbnail"
            onClick={() => onVideoClick(video, index)}
          >
            <iframe 
              src={video.url}
              className="thumbnail-preview"
              title={video.name}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              onError={(e) => {
                console.error('Error loading video:', video.url, e);
                e.target.style.display = 'none';
              }}
            />
            <div className="video-name">{video.location || video.name}</div>
            <button 
              className={`favorite-button ${video.isFavorite ? 'favorited' : ''}`}
              onClick={(e) => {
                e.stopPropagation();
                onToggleFavorite(index);
              }}
            >
              ★
            </button>
          </div>
        );
      })}
    </div>
  );
}

function LocationPopup({ isOpen, onClose, onSave, videoName }) {
  const [location, setLocation] = useState('');
  const [customName, setCustomName] = useState('');

  if (!isOpen) return null;

  return (
    <div className="popup-overlay">
      <div className="popup-content">
        <h2>Add Details for {videoName}</h2>
        <input
          type="text"
          placeholder="Enter video name"
          value={customName}
          onChange={(e) => setCustomName(e.target.value)}
          className="location-input"
        />
        <input
          type="text"
          placeholder="Enter video location"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          className="location-input"
        />
        <div className="popup-buttons">
          <button className="popup-button cancel" onClick={onClose}>Cancel</button>
          <button className="popup-button save" onClick={() => {
            onSave(location, customName);
            setLocation('');
            setCustomName('');
          }}>Save</button>
        </div>
      </div>
    </div>
  );
}

function VideoPlayer({ video, onClose }) {
  if (!video) return null;

  return (
    <div className="video-player-overlay">
      <div className="video-player-container">
        <button className="close-button" onClick={onClose}>×</button>
        <div className="video-player">
          <iframe
            src={video.url}
            title={video.name}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            className="video-frame"
          />
        </div>
        <div className="video-info">
          <h2>{video.location || video.name}</h2>
        </div>
      </div>
    </div>
  );
}

function VideoPage({ videos, onVideoClick, onToggleFavorite, showFavoritesOnly }) {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [showLocationPopup, setShowLocationPopup] = useState(false);
  const [pendingVideo, setPendingVideo] = useState(null);
  const fileInputRef = useRef(null);
  const location = useLocation();

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      try {
        const bucketName = process.env.REACT_APP_AWS_BUCKET_NAME;
        if (!bucketName) {
          throw new Error('Bucket name is not defined in environment variables');
        }

        // Generate a unique key for the video
        const key = `videos/${Date.now()}-${file.name}`;
        
        // Upload the file to S3
        const uploadParams = {
          Bucket: bucketName,
          Key: key,
          Body: file,
          ContentType: file.type
        };
        console.log('Upload params:', uploadParams);
        
        await s3.putObject(uploadParams).promise();
        
        // Get the signed URL for the uploaded video
        const urlParams = {
          Bucket: bucketName,
          Key: key,
          Expires: 3600 // URL expires in 1 hour
        };
        console.log('URL params:', urlParams);
        
        const url = await s3.getSignedUrlPromise('getObject', urlParams);
        
        const videoUrl = url;
        setPendingVideo({ url: videoUrl, name: file.name, key });
        setShowLocationPopup(true);
      } catch (error) {
        console.error('Error uploading video:', error);
      }
    }
  };

  const handleLocationSave = (location, customName) => {
    if (pendingVideo) {
      onVideoClick({ 
        ...pendingVideo, 
        location, 
        customName,
        isFavorite: false 
      }, videos.length);
      setPendingVideo(null);
    }
    setShowLocationPopup(false);
  };

  const handleCloseExpanded = () => {
    setSelectedVideo(null);
  };

  const navigateVideo = (direction) => {
    if (!selectedVideo) return;
    
    const currentVideos = showFavoritesOnly 
      ? videos.filter(video => video.isFavorite)
      : videos;
    
    const currentIndex = currentVideos.findIndex(v => v.url === selectedVideo.url);
    const newIndex = direction === 'next' 
      ? (currentIndex + 1) % currentVideos.length 
      : (currentIndex - 1 + currentVideos.length) % currentVideos.length;
    
    setSelectedVideo({ ...currentVideos[newIndex], index: newIndex });
  };

  return (
    <div className="page-content">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        style={{ display: 'none' }}
      />
      {!showFavoritesOnly && (
        <button className="upload-button" onClick={handleUploadClick}>Upload Video</button>
      )}
      
      <VideoGrid
        videos={videos}
        onVideoClick={setSelectedVideo}
        onToggleFavorite={onToggleFavorite}
        showFavoritesOnly={showFavoritesOnly}
      />

      <LocationPopup
        isOpen={showLocationPopup}
        onClose={() => setShowLocationPopup(false)}
        onSave={handleLocationSave}
        videoName={pendingVideo?.name || ''}
      />

      {selectedVideo && (
        <div className="expanded-video-container">
          <div className="expanded-video-overlay" onClick={handleCloseExpanded}></div>
          <div className="expanded-video-content">
            <div className="video-player-container">
              <button className="nav-video-button prev" onClick={() => navigateVideo('prev')}>←</button>
              <video 
                src={selectedVideo.url} 
                className="expanded-video"
                controls
                autoPlay
              />
              <button className="nav-video-button next" onClick={() => navigateVideo('next')}>→</button>
            </div>
            <div className="incidents-section">
              <h2>Incidents</h2>
              <div className="incidents-list">
                <div className="incident-item">
                  <div className="incident-time">00:01:23</div>
                  <div className="incident-description">Person detected</div>
                </div>
                <div className="incident-item">
                  <div className="incident-time">00:02:45</div>
                  <div className="incident-description">Motion detected</div>
                </div>
                <div className="incident-item">
                  <div className="incident-time">00:03:12</div>
                  <div className="incident-description">Object detected</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  const [videos, setVideos] = useState([]);

  const fetchUserVideos = async () => {
    try {
      const params = {
        Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
        Prefix: 'videos/'
      };

      const data = await s3.listObjectsV2(params).promise();
      
      const videos = await Promise.all(
        data.Contents.map(async (item) => {
          const url = await s3.getSignedUrlPromise('getObject', {
            Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
            Key: item.Key,
            Expires: 3600
          });
          
          return {
            url,
            name: item.Key.split('/').pop(),
            key: item.Key
          };
        })
      );

      setVideos(videos);
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  useEffect(() => {
    fetchUserVideos();
  }, []);

  const toggleFavorite = (videoIndex) => {
    const updatedVideos = [...videos];
    updatedVideos[videoIndex].isFavorite = !updatedVideos[videoIndex].isFavorite;
    setVideos(updatedVideos);
  };

  return (
    <div className="App">
      <NavBar />
      <Routes>
        <Route path="/" element={
          <VideoPage
            videos={videos}
            onVideoClick={(video, index) => setVideos([...videos, video])}
            onToggleFavorite={toggleFavorite}
            showFavoritesOnly={false}
          />
        } />
        <Route path="/favorites" element={
          <VideoPage
            videos={videos}
            onVideoClick={(video, index) => setVideos([...videos, video])}
            onToggleFavorite={toggleFavorite}
            showFavoritesOnly={true}
          />
        } />
        <Route path="/realtime" element={<div className="page-content">Realtime Page</div>} />
      </Routes>
    </div>
  );
}

export default App;
