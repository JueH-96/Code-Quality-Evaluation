import math # Not strictly necessary for this problem, but good habit
from typing import List # Required for type hinting

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        """
        Finds the button index that corresponds to the longest press duration.
        The duration for the first press is its timestamp. The duration for subsequent
        presses is the time difference between the current press and the previous press.
        If there's a tie in duration, returns the smallest button index among the ties.

        Args:
            events: A list of events, where each event is [index_i, time_i].
                    The list is sorted by time_i in increasing order. 
                    Constraints guarantee events is not empty.

        Returns:
            The index (index_i) of the button that had the longest press time,
            with ties broken by choosing the smaller index.
        """
        
        # Constraints state 1 <= events.length, so events[0] is always accessible.
        
        # Initialize variables based on the first event.
        # The time taken for the first button press is its timestamp, as the duration 
        # is implicitly measured from time 0.
        max_duration = events[0][1] 
        
        # The index of the button associated with this initial maximum duration.
        button_index_with_max_duration = events[0][0]
        
        # Keep track of the timestamp of the previous event to calculate durations
        # for subsequent events. Initialize with the time of the first event.
        previous_event_time = events[0][1]

        # Iterate through the events starting from the second one (index 1).
        # The loop runs from i = 1 to len(events) - 1.
        for i in range(1, len(events)):
            # Get the index and time for the current event.
            current_button_index = events[i][0]
            current_event_time = events[i][1]

            # Calculate the duration for the current button press.
            # This is the time elapsed since the previous button press.
            current_duration = current_event_time - previous_event_time

            # Check if the current duration is greater than the maximum duration found so far.
            if current_duration > max_duration:
                # If it is, update the maximum duration and store the index of the current button.
                max_duration = current_duration
                button_index_with_max_duration = current_button_index
            elif current_duration == max_duration:
                # If the current duration is equal to the maximum duration, we need to apply 
                # the tie-breaking rule: choose the button with the smaller index.
                # Update the result index only if the current button's index is smaller.
                button_index_with_max_duration = min(button_index_with_max_duration, current_button_index)

            # Update the time of the previous event to the current event's time, 
            # preparing for the next iteration's duration calculation.
            previous_event_time = current_event_time

        # After iterating through all events, button_index_with_max_duration holds the index 
        # of the button that resulted in the longest duration, respecting the tie-breaking rule.
        return button_index_with_max_duration