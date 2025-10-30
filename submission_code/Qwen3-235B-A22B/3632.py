from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = -1
        best_index = -1
        previous_time = 0
        
        for index, current_time in events:
            delta = current_time - previous_time
            if delta > max_time:
                max_time = delta
                best_index = index
            elif delta == max_time:
                if index < best_index:
                    best_index = index
            previous_time = current_time
        
        return best_index