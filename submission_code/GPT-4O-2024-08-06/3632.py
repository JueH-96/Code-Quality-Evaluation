class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # Initialize variables to track the longest time and the corresponding button index
        longest_time = events[0][1]  # The first button press time is the time itself
        button_index = events[0][0]  # The index of the first button pressed
        
        # Iterate over the events starting from the second event
        for i in range(1, len(events)):
            # Calculate the time taken for the current button press
            current_time = events[i][1] - events[i-1][1]
            
            # Check if the current time is greater than the longest time found so far
            if current_time > longest_time:
                longest_time = current_time
                button_index = events[i][0]
            # If the current time is equal to the longest time, choose the smaller index
            elif current_time == longest_time:
                button_index = min(button_index, events[i][0])
        
        return button_index