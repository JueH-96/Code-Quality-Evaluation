class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = 0
        button_with_longest_time = events[0][0]
        
        # Initialize the previous time with the time of the first event
        previous_time = events[0][1]
        
        for index, time in events[1:]:
            # Calculate the time difference between the current and previous event
            time_diff = time - previous_time
            
            # Update the previous time to the current time
            previous_time = time
            
            # Check if the current time difference is greater than the max found so far
            if time_diff > max_time:
                max_time = time_diff
                button_with_longest_time = index
            elif time_diff == max_time:
                # If the time difference is the same, choose the smaller index
                button_with_longest_time = min(button_with_longest_time, index)
        
        return button_with_longest_time