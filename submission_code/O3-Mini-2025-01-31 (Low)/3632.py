class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # Initialize with the first event's duration and button index.
        max_duration = events[0][1]  # Duration for first press.
        result_button = events[0][0]
        
        # Iterate over subsequent events to compute durations.
        for i in range(1, len(events)):
            button, time = events[i]
            prev_time = events[i-1][1]
            duration = time - prev_time
            
            # Check if this duration is longer than the current max_duration.
            if duration > max_duration:
                max_duration = duration
                result_button = button
            # In case of a tie, the one with smallest index is chosen.
            elif duration == max_duration:
                result_button = min(result_button, button)
                
        return result_button