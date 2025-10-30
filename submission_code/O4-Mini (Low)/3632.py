from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        prev_time = 0
        max_duration = -1
        result_index = None
        
        for idx, t in events:
            duration = t - prev_time
            if duration > max_duration or (duration == max_duration and (result_index is None or idx < result_index)):
                max_duration = duration
                result_index = idx
            prev_time = t
        
        return result_index