class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = events[0][1]  # Time for first button
        result_button = events[0][0]  # Index of first button
        
        # Check remaining events
        for i in range(1, len(events)):
            current_index = events[i][0]
            current_time = events[i][1]
            prev_time = events[i-1][1]
            
            time_taken = current_time - prev_time
            
            # Update if we found a longer time, or same time with smaller index
            if time_taken > max_time or (time_taken == max_time and current_index < result_button):
                max_time = time_taken
                result_button = current_index
        
        return result_button