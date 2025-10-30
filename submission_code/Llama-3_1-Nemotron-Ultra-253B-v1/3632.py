from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        time_sums = {}
        prev_time = None
        for event in events:
            button, current_time = event
            if prev_time is None:
                time_diff = current_time
            else:
                time_diff = current_time - prev_time
            if button in time_sums:
                time_sums[button] += time_diff
            else:
                time_sums[button] = time_diff
            prev_time = current_time
        
        max_time = -1
        result_button = None
        for button in time_sums:
            total = time_sums[button]
            if total > max_time or (total == max_time and button < result_button):
                max_time = total
                result_button = button
        return result_button