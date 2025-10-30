from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_duration = -1
        result_index = -1
        prev_time = None
        
        for event in events:
            index, time = event
            if prev_time is None:
                duration = time
            else:
                duration = time - prev_time
            prev_time = time
            
            if duration > max_duration:
                max_duration = duration
                result_index = index
            elif duration == max_duration:
                if index < result_index:
                    result_index = index
        
        return result_index