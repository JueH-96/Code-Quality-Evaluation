class Solution:
    def findLatestTime(self, s: str) -> str:
        # Split the input into hour and minute parts
        hour_part, minute_part = s.split(':')
        
        max_hour = None
        # Check all possible hours from 11 down to 0
        for hour in range(11, -1, -1):
            hour_str = f"{hour:02d}"
            # Check if this hour can be formed by replacing '?' in the hour_part
            valid = True
            for h1, h2 in zip(hour_part, hour_str):
                if h1 != '?' and h1 != h2:
                    valid = False
                    break
            if valid:
                max_hour = hour_str
                break
        
        max_minute = None
        # Check all possible minutes from 59 down to 0
        for minute in range(59, -1, -1):
            minute_str = f"{minute:02d}"
            # Check if this minute can be formed by replacing '?' in the minute_part
            valid = True
            for m1, m2 in zip(minute_part, minute_str):
                if m1 != '?' and m1 != m2:
                    valid = False
                    break
            if valid:
                max_minute = minute_str
                break
        
        return f"{max_hour}:{max_minute}"