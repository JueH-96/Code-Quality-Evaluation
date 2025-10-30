from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # Handle the first event
        prev_time = events[0][1]
        max_delta = prev_time  # Delta for the first event is its time
        result_index = events[0][0]
        
        # Iterate through the remaining events
        for event in events[1:]:
            current_time = event[1]
            delta = current_time - prev_time
            if delta > max_delta:
                max_delta = delta
                result_index = event[0]
            elif delta == max_delta:
                if event[0] < result_index:
                    result_index = event[0]
            # Update previous time for the next iteration
            prev_time = current_time
        
        return result_index