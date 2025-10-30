from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # Initial button press (first event)
        max_time = events[0][1]
        answer_index = events[0][0]
        
        # Iterate through the remaining events
        for i in range(1, len(events)):
            # Time taken for this press
            duration = events[i][1] - events[i - 1][1]
            button_idx = events[i][0]
            
            # Update if we find a longer duration, or same duration with a smaller index
            if duration > max_time or (duration == max_time and button_idx < answer_index):
                max_time = duration
                answer_index = button_idx
                
        return answer_index