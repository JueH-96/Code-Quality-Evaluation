from typing import List

class Solution:
  def buttonWithLongestTime(self, events: List[List[int]]) -> int:
    # max_duration will store the maximum time found so far for a button press.
    # Initialize to -1, as actual durations will be non-negative (first duration is events[0][1] >= 1).
    max_duration = -1
    
    # result_button_index will store the index of the button corresponding to max_duration.
    # Initialize to -1; it will be set by the first event.
    result_button_index = -1 
    
    # previous_event_time stores the time of the previous event.
    # Initialize to 0, so for the first event, duration = events[0][1] - 0 = events[0][1].
    previous_event_time = 0
    
    # Iterate through each event
    for i in range(len(events)):
        current_button_index = events[i][0]
        current_event_time = events[i][1]
        
        # Calculate the time taken for this specific button press
        current_duration = current_event_time - previous_event_time
        
        # Check if this duration is the longest found so far
        if current_duration > max_duration:
            max_duration = current_duration
            result_button_index = current_button_index
        elif current_duration == max_duration:
            # If this duration is equal to the current max_duration,
            # apply the tie-breaking rule: "return the button with the smallest index."
            # We update result_button_index only if current_button_index is smaller.
            result_button_index = min(result_button_index, current_button_index)
        
        # Update previous_event_time for the next iteration's calculation
        previous_event_time = current_event_time
            
    return result_button_index