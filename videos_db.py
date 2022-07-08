import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import List, Optional


@dataclass()
class VideoData:
    id: int
    title: str
    thumb: str
    source: str
    subtitle: str = ""


class BaseVideosDB(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_videos(self) -> List[VideoData]:
        pass

    @abstractmethod
    def add_video(self, video_data: VideoData) -> None:
        pass

    @abstractmethod
    def get_video_by_id(self, video_id: str) -> Optional[VideoData]:
        pass


class VideosJsonDB(BaseVideosDB):

    def __init__(self, json_path):
        self.json_path = json_path

    def get_videos(self):
        with open(self.json_path, 'r') as f:
            videos = json.load(f)
        return [VideoData(**video) for video in videos]

    def add_video(self, video_data: VideoData):
        videos = self.get_videos()
        videos.append(asdict(video_data))
        with open(self.json_path, 'w') as f:
            json.dump(videos, f)

    def get_video_by_id(self, video_id: str):
        videos = self.get_videos()
        matches = filter(lambda x: str(x.id) == str(video_id), videos)
        if matches:
            return next(matches)
        return None