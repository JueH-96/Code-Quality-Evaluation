class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        if not events:
            return -1  # or an appropriate value indicating no events
        
        # Initialize with the first event
        max_time = events[0][1]
        result_index = events[0][0]
        previous_time = events[0][1]
        
        # Iterate from the second event onwards
        for i in range(1, len(events)):
            current_index, current_time = events[i]
            time_difference = current_time - previous_time
            if time_difference > max_time:
                max_time = time_difference
                result_index = current_index
            elif time_difference == max_time:
                result_index = min(result_index, current_index)
            previous_time = current_time
        
        return result_index