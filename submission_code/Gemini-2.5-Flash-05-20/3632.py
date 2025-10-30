from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # According to constraints, events.length will be at least 1,
        # so we can safely access events[0].

        # Initialize max_time_taken with the time of the first event,
        # as per problem definition ("The time for the first button is simply the time at which it was pressed").
        max_time_taken = events[0][1]
        
        # Initialize result_button_index with the index of the first button.
        result_button_index = events[0][0]
        
        # previous_time keeps track of the time of the last processed event.
        # It's used to calculate the duration of the current event.
        previous_time = events[0][1]

        # Iterate through the events starting from the second event (index 1).
        # The first event has already been handled for initialization.
        for i in range(1, len(events)):
            current_index = events[i][0]
            current_time = events[i][1]

            # Calculate the time taken for the current button press.
            # This is the difference between the current event's time and the previous event's time.
            current_time_taken = current_time - previous_time

            # Compare the current time taken with the maximum found so far.
            if current_time_taken > max_time_taken:
                # If a new maximum time is found, update both max_time_taken and result_button_index.
                max_time_taken = current_time_taken
                result_button_index = current_index
            elif current_time_taken == max_time_taken:
                # If the current time taken is equal to the maximum, apply the tie-breaking rule:
                # "return the button with the smallest index".
                # If the current button's index is smaller than the one currently stored, update it.
                if current_index < result_button_index:
                    result_button_index = current_index
            
            # Update previous_time to the current_time for the next iteration.
            previous_time = current_time
            
        return result_button_index