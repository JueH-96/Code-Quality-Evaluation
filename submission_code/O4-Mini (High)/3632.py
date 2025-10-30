from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        prev_time = 0
        max_time = -1
        result = 0
        
        for idx, t in events:
            duration = t - prev_time
            # update if we found a strictly longer duration
            # or same duration but smaller button index
            if duration > max_time or (duration == max_time and idx < result):
                max_time = duration
                result = idx
            prev_time = t
        
        return result