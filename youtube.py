from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse as urlparse

# Function to extract video id from url
def extract_video_id(url):
    # Parse URL
    url_data = urlparse.urlparse(url)
    # Extract video id
    video_id = urlparse.parse_qs(url_data.query)['v'][0]
    return video_id

# Prompt for YouTube video URL
url = input("Enter YouTube video URL: ")

# Extract video id
video_id = extract_video_id(url)

# Fetch the captions
try:
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_generated_transcript(['en'])
    captions = transcript.fetch()

    # Open a text file in write mode
    with open('captions.txt', 'w') as f:
        for caption in captions:
            print(caption['text'])
            # Write the caption to the text file
            f.write(caption['text'] + '\n')
except:
    print("An error occurred while fetching the captions.")
