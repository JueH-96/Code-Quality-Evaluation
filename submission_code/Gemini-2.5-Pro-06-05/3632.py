from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        """
        Finds the button index that had the longest press time.

        The time taken for a button press is the difference in time between
        consecutive events. For the first event, the time taken is its timestamp.
        If there's a tie for the longest time, the button with the smallest
        index is returned.

        Args:
            events: A list of events, where each event is [index, time],
                    sorted by time in increasing order.

        Returns:
            The index of the button with the longest press time.
        """
        
        # According to the problem description:
        # - The input `events` is sorted by time.
        # - The time for the first button is its timestamp.
        # - The time for subsequent buttons is the difference from the previous one.
        # - If there's a tie in time, the button with the smaller index wins.

        # Constraints guarantee at least one event, so events[0] is safe to access.
        # Initialize with the first event's data. The duration for the first event
        # is simply its timestamp.
        max_duration = events[0][1]
        result_button = events[0][0]

        # Iterate through the rest of the events, starting from the second one.
        for i in range(1, len(events)):
            # The button being pressed in this event.
            current_button = events[i][0]
            
            # The time duration for this press is the difference from the previous press's time.
            duration = events[i][1] - events[i-1][1]

            # Check if we have a new longest duration.
            if duration > max_duration:
                max_duration = duration
                result_button = current_button
            # Handle the tie-breaking case: if durations are equal, choose the
            # button with the smaller index.
            elif duration == max_duration:
                result_button = min(result_button, current_button)
        
        return result_button