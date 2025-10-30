from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # Initialize with the time of the first button press
        prev_time = events[0][1]
        max_duration = prev_time
        answer_index = events[0][0]
        
        # Iterate over subsequent events to compute durations
        for button_index, press_time in events[1:]:
            duration = press_time - prev_time
            # Update if a longer duration is found, or same duration with a smaller index
            if duration > max_duration or (duration == max_duration and button_index < answer_index):
                max_duration = duration
                answer_index = button_index
            prev_time = press_time
        
        return answer_index