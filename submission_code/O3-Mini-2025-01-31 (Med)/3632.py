from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # The first event's duration is its own timestamp.
        max_time = events[0][1]  
        button_index = events[0][0]
        
        # Iterate over the events starting from second event
        for i in range(1, len(events)):
            # The duration for the current event is the difference in time from previous event
            duration = events[i][1] - events[i - 1][1]
            
            # Check if the current duration is greater than the max time found
            if duration > max_time:
                max_time = duration
                button_index = events[i][0]
            # If the duration is equal, choose the button with the smallest index
            elif duration == max_time and events[i][0] < button_index:
                button_index = events[i][0]
                
        return button_index