from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        longest_time = 0
        longest_button = -1
        previous_time = 0
        
        for i, (index, time) in enumerate(events):
            if i == 0:
                # For the first button press, the time taken is just the time of the first event
                time_taken = time
            else:
                # Time taken is the difference between the current time and the previous time
                time_taken = time - previous_time
            
            # Update the previous time for the next iteration
            previous_time = time
            
            # Check if we found a longer time or if it's the same but a smaller index
            if time_taken > longest_time or (time_taken == longest_time and index < longest_button):
                longest_time = time_taken
                longest_button = index
        
        return longest_button