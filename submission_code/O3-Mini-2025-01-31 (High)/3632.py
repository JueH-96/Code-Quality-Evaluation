from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # The duration for the first button press is the time of the first event.
        max_duration = events[0][1]
        result_button = events[0][0]
        
        # For each subsequent event, the duration is the difference in time
        for i in range(1, len(events)):
            duration = events[i][1] - events[i - 1][1]
            # Update if a longer duration is found, or if equal, choose the smaller button index.
            if duration > max_duration:
                max_duration = duration
                result_button = events[i][0]
            elif duration == max_duration and events[i][0] < result_button:
                result_button = events[i][0]
        
        return result_button