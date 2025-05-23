import re
from typing import List, Dict, Any

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def remove_timestamps(lines: List[str]) -> List[str]:
    pattern = re.compile(r'^\[.*\]')
    return [pattern.sub('', line).strip() for line in lines]

def format_output(
    transcript: List[str],
    format_type: str = "txt",
    include_word_timestamps: bool = False
) -> str:
    if format_type == "txt":
        return "\n".join(transcript)
    elif format_type == "srt":
        return "\n\n".join([f"{i+1}\n{line}" for i, line in enumerate(transcript)])
    elif format_type == "vtt":
        return "WEBVTT\n\n" + "\n\n".join([f"{i+1}\n{line.replace(' --> ', ' --> ')}" 
                                         for i, line in enumerate(transcript)])
    elif format_type == "json":
        import json
        return json.dumps([{"text": line} for line in transcript], indent=2, ensure_ascii=False)
    return "\n".join(transcript)
