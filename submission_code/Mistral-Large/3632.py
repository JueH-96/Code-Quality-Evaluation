from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = 0
        max_index = events[0][0]

        for i in range(1, len(events)):
            time_taken = events[i][1] - events[i-1][1]
            if time_taken > max_time or (time_taken == max_time and events[i][0] < max_index):
                max_time = time_taken
                max_index = events[i][0]

        return max_index