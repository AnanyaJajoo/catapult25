from pymavi import MaviClient
import os
import dotenv
from pymavi.exceptions import MaviAPIError
from collections import defaultdict
import datetime
import json

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(dotenv_path)

API_KEY = os.getenv("MAVI_API_KEY")

def format_timestamp(seconds):
    """Convert seconds to a human-readable timestamp."""
    return str(datetime.timedelta(seconds=int(seconds)))

def get_top_fragments_by_query(client, search_query, top_n=5):
    """
    Search across all videos for a specific query and return the top timestamp fragments.
    
    Args:
        client: MaviClient instance
        search_query: Query string to search for
        top_n: Number of top results to return per video
        
    Returns:
        Dictionary mapping normalized video names to lists of top fragments
    """
    try:
        # Get all video IDs
        video_metadata = client.search_video_metadata()
        video_ids = list(video_metadata.keys())
        
        if not video_ids:
            print("No videos found in your Mavi account.")
            return {}
        
        # Search for the query across all videos
        search_results = client.search_key_clip(
            video_ids=video_ids, 
            search_query=search_query
        )
        
        return process_search_results(search_results, top_n)
        
    except MaviAPIError as e:
        print(f"Error searching videos: {e}")
        return {}

def process_search_results(results, top_n=5):
    """
    Process raw search results and extract top fragments per video.
    
    This function can be used independently with saved results.
    """
    # Normalize video names by removing path variations
    normalized_results = []
    for item in results:
        video_name = item['videoName']
        # Extract just the filename without path
        if '/' in video_name:
            video_name = video_name.split('/')[-1]
        elif '\\' in video_name:
            video_name = video_name.split('\\')[-1]
        elif './' in video_name:
            video_name = video_name[2:]
        
        normalized_results.append({
            'videoId': item['videoNo'],
            'videoName': video_name,
            'start': int(item['fragmentStartTime']),
            'end': int(item['fragmentEndTime']),
            'duration': int(item['duration']),
            'fragment_length': int(item['fragmentEndTime']) - int(item['fragmentStartTime'])
        })
    
    # Group by normalized video name
    videos_dict = defaultdict(list)
    for item in normalized_results:
        videos_dict[item['videoName']].append(item)
    
    # For each video, sort fragments by length (longer fragments may be more relevant)
    # and take the top N
    top_results = {}
    for video_name, fragments in videos_dict.items():
        # Sort by fragment length (descending)
        sorted_fragments = sorted(fragments, key=lambda x: x['fragment_length'], reverse=True)
        
        # Remove duplicates (fragments that heavily overlap)
        unique_fragments = []
        for fragment in sorted_fragments:
            # Check if this fragment heavily overlaps with any we've already selected
            is_duplicate = False
            for selected in unique_fragments:
                # Calculate overlap percentage
                overlap_start = max(fragment['start'], selected['start'])
                overlap_end = min(fragment['end'], selected['end'])
                
                if overlap_end > overlap_start:
                    overlap_length = overlap_end - overlap_start
                    shorter_length = min(fragment['fragment_length'], selected['fragment_length'])
                    
                    # If overlap is more than 60% of the shorter fragment, consider it a duplicate
                    if overlap_length / shorter_length > 0.6:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_fragments.append(fragment)
            
            # Stop once we have enough unique fragments
            if len(unique_fragments) >= top_n:
                break
        
        # Format the results
        formatted_fragments = []
        for i, frag in enumerate(unique_fragments[:top_n], 1):
            formatted_fragments.append({
                'rank': i,
                'start_time': frag['start'],
                'end_time': frag['end'],
                'formatted_start': format_timestamp(frag['start']),
                'formatted_end': format_timestamp(frag['end']),
                'duration': frag['fragment_length'],
                'video_id': frag['videoId']
            })
        
        top_results[video_name] = formatted_fragments
    
    return top_results

def get_top_overall_fragments(client, search_query, top_n=3):
    """
    Search across all videos for a specific query using a two-step process:
    1. First find relevant videos using search_video
    2. Then search for key clips within those videos
    
    Args:
        client: MaviClient instance
        search_query: Query string to search for
        top_n: Number of top results to return across all videos
        
    Returns:
        List of top fragments across all videos
    """
    try:
        # Step 1: Find relevant videos first
        relevant_videos = client.search_video(search_query)
        
        if not relevant_videos:
            print(f"No videos found matching '{search_query}'.")
            return []
        
        # Extract video IDs from the relevant videos
        video_ids = list(relevant_videos.keys())
        print(f"Found {len(video_ids)} potentially relevant videos.")
        video_ids = video_ids[:5]  # Limit to top 3 videos for further search
        
        # Step 2: Search for specific clips within those videos
        search_results = client.search_key_clip(
            video_ids=video_ids, 
            search_query=search_query
        )
        
        if not search_results:
            print("No specific clips found within the relevant videos.")
            return []
            
        print(f"Found {len(search_results)} potentially relevant clips.")
        
        # Process all fragments regardless of video
        normalized_results = []
        for item in search_results:
            video_name = item['videoName']
            # Extract just the filename without path
            if '/' in video_name:
                video_name = video_name.split('/')[-1]
            elif '\\' in video_name:
                video_name = video_name.split('\\')[-1]
            elif './' in video_name:
                video_name = video_name[2:]
            
            fragment_length = int(item['fragmentEndTime']) - int(item['fragmentStartTime'])
            
            normalized_results.append({
                'videoId': item['videoNo'],
                'videoName': video_name,
                'start': int(item['fragmentStartTime']),
                'end': int(item['fragmentEndTime']),
                'duration': int(item['duration']),
                'fragment_length': fragment_length
            })
        
        # Sort all fragments by length (descending)
        sorted_fragments = sorted(normalized_results, key=lambda x: x['fragment_length'], reverse=True)
        
        # Remove duplicates (fragments that heavily overlap or are from the same part of the same video)
        unique_fragments = []
        for fragment in sorted_fragments:
            # Check if this fragment heavily overlaps with any we've already selected
            is_duplicate = False
            for selected in unique_fragments:
                # Only check for overlap if from the same video
                if fragment['videoName'] == selected['videoName']:
                    # Calculate overlap percentage
                    overlap_start = max(fragment['start'], selected['start'])
                    overlap_end = min(fragment['end'], selected['end'])
                    
                    if overlap_end > overlap_start:
                        overlap_length = overlap_end - overlap_start
                        shorter_length = min(fragment['fragment_length'], selected['fragment_length'])
                        
                        # If overlap is more than 60% of the shorter fragment, consider it a duplicate
                        if overlap_length / shorter_length > 0.6:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_fragments.append(fragment)
            
            # Stop once we have enough unique fragments
            if len(unique_fragments) >= top_n:
                break
        
        # Format the results
        top_fragments = []
        for i, frag in enumerate(unique_fragments[:top_n], 1):
            top_fragments.append({
                'rank': i,
                'video_name': frag['videoName'],
                'start_time': frag['start'],
                'end_time': frag['end'],
                'formatted_start': format_timestamp(frag['start']),
                'formatted_end': format_timestamp(frag['end']),
                'duration': frag['fragment_length'],
                'video_id': frag['videoId']
            })
        
        return top_fragments
        
    except MaviAPIError as e:
        print(f"Error searching videos: {e}")
        return []

def upload_video(client, video_path):
    """Upload a video to Mavi and return its ID."""
    try:
        print(f"Uploading video: {video_path}...")
        video_id = client.upload_video(video_path)
        print(f"Video uploaded successfully. Video ID: {video_id}")
        return video_id
    except MaviAPIError as e:
        print(f"Error uploading video: {e}")
        return None
    
def delete_all_videos(client):
    client.delete_video(list(client.search_video_metadata().keys()))
    print("All videos deleted successfully.")

def main():
    client = MaviClient(API_KEY)
    
    # Example video upload (uncomment to use)
    # video_path = "long_videos/video_11.mp4"
    # delete_all_videos(client)  # Uncomment to delete all videos before uploading
    # video_id = upload_video(client, video_path)
    
    # Upload all videos in the directory (uncomment to use)
    # video_dir = "long_videos"
    # for video_file in os.listdir(video_dir):
    #     video_path = os.path.join(video_dir, video_file)
    #     upload_video(client, video_path)
    
    # Example usage
    query = "Find me videos of a man in an orange shirt climbing the stairs"
    print(f"Searching for '{query}' across all videos...")
        
    # print(client.search_video_metadata(video_status="PARSE", num_results=20))
    
    # Get top 3 fragments across all videos
    top_fragments = get_top_overall_fragments(client, search_query=query, top_n=3)
    
    if not top_fragments:
        print("No results found.")
        return
    
    # Print the results in a nice format
    print(f"\nTop 3 fragments overall for query: '{query}'")
    print("-" * 60)
    
    for fragment in top_fragments:
        print(f"\n🏆 Rank #{fragment['rank']}: {fragment['video_name']}")
        print(f"  Time: {fragment['formatted_start']} to {fragment['formatted_end']} " 
              f"(Duration: {format_timestamp(fragment['duration'])})")
        print(f"  Video ID: {fragment['video_id']}")

if __name__ == "__main__":
    main()