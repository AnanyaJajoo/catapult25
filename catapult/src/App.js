import { useState, useRef, useEffect } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { s3 } from './aws-config';
import './App.css';

function NavBar() {
  const location = useLocation();
  
  return (
    <nav className="nav-bar">
      <Link to="/" className="logo">
        <span>Spectra</span>
      </Link>
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
          <input type="text" placeholder="Search videos..." className="search-input" />
        </div>
      </div>
    </nav>
  );
}

function VideoGrid({ videos, onVideoClick, onToggleFavorite, showFavoritesOnly = false, isPlaying = false }) {
  const displayedVideos = showFavoritesOnly 
    ? videos.filter(video => video.isFavorite)
    : videos;

  return (
    <div className="video-grid">
      {displayedVideos.map((video, index) => (
        <div 
          key={index} 
          className="video-thumbnail"
          onClick={() => onVideoClick(video, index)}
        >
          <video 
            src={video.url}
            className="thumbnail-preview"
            title={video.customName || video.name}
            muted
            autoPlay
            loop
            playsInline
            style={{ pointerEvents: 'none' }}
          />
          <div className="video-info-container">
            <div className="video-name">{video.customName || video.name}</div>
            <div className="video-location">{video.location || 'No location specified'}</div>
          </div>
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
      ))}
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

function VideoPlayer({ video, onClose, onNext, onPrev, hasNext, hasPrev }) {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef(null);

  // Mock incidents data - replace with your actual data
  const incidents = [
    { time: 30, description: "Person detected" },
    { time: 120, description: "Motion detected" },
    { time: 180, description: "Object detected" },
    { time: 240, description: "Person detected" },
  ];

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleTimestampClick = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  if (!video) return null;

  return (
    <div className="video-player-overlay">
      <div className="video-player-container">
        <button className="close-button" onClick={onClose}>×</button>
        <div className="video-player-content">
          <div className="video-player-main">
            <div className="video-player">
              <video
                ref={videoRef}
                src={video.url}
                title={video.name}
                controls
                autoPlay
                className="video-frame"
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
              />
            </div>
            <div className="video-info">
              <h2>{video.location || video.name}</h2>
              <div className="video-time">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>
            <div className="video-navigation">
              <button 
                className="nav-button prev" 
                onClick={onPrev}
                disabled={!hasPrev}
              >
                ← Previous
              </button>
              <button 
                className="nav-button next" 
                onClick={onNext}
                disabled={!hasNext}
              >
                Next →
              </button>
            </div>
          </div>
          <div className="video-timeline">
            <h3>Timeline</h3>
            <div className="incidents-list">
              {incidents.map((incident, index) => (
                <div 
                  key={index}
                  className={`incident-item ${currentTime >= incident.time && currentTime < incident.time + 5 ? 'active' : ''}`}
                  onClick={() => handleTimestampClick(incident.time)}
                >
                  <div className="incident-time">{formatTime(incident.time)}</div>
                  <div className="incident-description">{incident.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function VideoPage({ onVideoClick, onToggleFavorite, showFavoritesOnly }) {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [showLocationPopup, setShowLocationPopup] = useState(false);
  const [pendingVideo, setPendingVideo] = useState(null);
  const [videos, setVideos] = useState([]);
  const fileInputRef = useRef(null);
  const location = useLocation();

  useEffect(() => {
    fetchUserVideos();
  }, []);

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

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      try {
        const key = `videos/${Date.now()}-${file.name}`;
        
        // Upload the file to S3
        await s3.putObject({
          Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
          Key: key,
          Body: file,
          ContentType: file.type
        }).promise();
        
        // Get the signed URL for the uploaded video
        const url = await s3.getSignedUrlPromise('getObject', {
          Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
          Key: key,
          Expires: 3600
        });
        
        // Add the new video to the state
        const newVideo = {
          url,
          name: file.name,
          key: key,
          isFavorite: false
        };
        
        setVideos(prevVideos => [...prevVideos, newVideo]);
        setPendingVideo(newVideo);
        setShowLocationPopup(true);
      } catch (error) {
        console.error('Error uploading video:', error);
      }
    }
  };

  const handleLocationSave = (location, customName) => {
    if (pendingVideo) {
      setVideos(prevVideos => 
        prevVideos.map(video => 
          video.key === pendingVideo.key 
            ? { ...video, location, customName }
            : video
        )
      );
      setPendingVideo(null);
    }
    setShowLocationPopup(false);
  };

  const handleVideoClick = (video, index) => {
    setSelectedVideo({ ...video, index });
  };

  const handleCloseVideo = () => {
    setSelectedVideo(null);
  };

  const handleNextVideo = () => {
    if (selectedVideo && selectedVideo.index < videos.length - 1) {
      setSelectedVideo({ ...videos[selectedVideo.index + 1], index: selectedVideo.index + 1 });
    }
  };

  const handlePrevVideo = () => {
    if (selectedVideo && selectedVideo.index > 0) {
      setSelectedVideo({ ...videos[selectedVideo.index - 1], index: selectedVideo.index - 1 });
    }
  };

  const handleToggleFavorite = (index) => {
    setVideos(prevVideos => 
      prevVideos.map((video, i) => 
        i === index ? { ...video, isFavorite: !video.isFavorite } : video
      )
    );
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
        onVideoClick={handleVideoClick}
        onToggleFavorite={handleToggleFavorite}
        showFavoritesOnly={showFavoritesOnly}
        isPlaying={!selectedVideo}
      />

      {selectedVideo && (
        <VideoPlayer
          video={selectedVideo}
          onClose={handleCloseVideo}
          onNext={handleNextVideo}
          onPrev={handlePrevVideo}
          hasNext={selectedVideo.index < videos.length - 1}
          hasPrev={selectedVideo.index > 0}
        />
      )}

      <LocationPopup
        isOpen={showLocationPopup}
        onClose={() => setShowLocationPopup(false)}
        onSave={handleLocationSave}
        videoName={pendingVideo?.name || ''}
      />
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
