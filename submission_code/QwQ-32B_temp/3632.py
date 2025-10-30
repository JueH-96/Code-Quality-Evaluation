from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        if not events:
            return -1  # According to constraints, this case won't occur
        
        max_time = events[0][1]
        best_index = events[0][0]
        
        for i in range(1, len(events)):
            time_diff = events[i][1] - events[i-1][1]
            if time_diff > max_time:
                max_time = time_diff
                best_index = events[i][0]
            elif time_diff == max_time:
                if events[i][0] < best_index:
                    best_index = events[i][0]
        
        return best_index