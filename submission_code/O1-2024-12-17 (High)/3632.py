from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = 0
        button_with_max_time = events[0][0]
        
        # Time for the first button
        prev_time = events[0][1]
        max_time = prev_time
        
        for i in range(1, len(events)):
            curr_index, curr_time = events[i]
            time_taken = curr_time - prev_time
            
            if time_taken > max_time:
                max_time = time_taken
                button_with_max_time = curr_index
            elif time_taken == max_time and curr_index < button_with_max_time:
                button_with_max_time = curr_index
            
            prev_time = curr_time
        
        return button_with_max_time