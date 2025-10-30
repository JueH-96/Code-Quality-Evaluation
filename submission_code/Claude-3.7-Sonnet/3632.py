class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # Get the first button's index and time
        index, time = events[0]
        max_time = time  # Time for first button is its time
        max_time_index = index  # Initially, button with max time is the first one
        
        # Iterate through remaining events
        for i in range(1, len(events)):
            current_index, current_time = events[i]
            time_taken = current_time - events[i-1][1]  # Time taken for current button
            
            # Update if we found a longer time or equal time with smaller index
            if time_taken > max_time:
                max_time = time_taken
                max_time_index = current_index
            elif time_taken == max_time and current_index < max_time_index:
                max_time_index = current_index
        
        return max_time_index