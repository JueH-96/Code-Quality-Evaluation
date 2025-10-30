from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # Handle the case with only one event
        if len(events) == 1:
            return events[0][0]
        
        # Initialize max_time with the time of the first event
        max_time = events[0][1]
        result_index = events[0][0]
        prev_time = events[0][1]
        
        # Iterate through the events starting from the second one
        for i in range(1, len(events)):
            current_time = events[i][1]
            time_taken = current_time - prev_time
            if time_taken > max_time:
                max_time = time_taken
                result_index = events[i][0]
            elif time_taken == max_time:
                if events[i][0] < result_index:
                    result_index = events[i][0]
            prev_time = current_time
        
        return result_index