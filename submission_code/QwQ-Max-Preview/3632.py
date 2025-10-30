from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_times = {}
        prev_time = 0
        for i, (idx, time) in enumerate(events):
            if i == 0:
                taken = time
            else:
                taken = time - prev_time
            # Update the maximum time for the current index
            if idx in max_times:
                if taken > max_times[idx]:
                    max_times[idx] = taken
            else:
                max_times[idx] = taken
            prev_time = time
        
        # Find the index with the maximum time, choosing the smallest index in case of ties
        max_time = -1
        best_index = -1
        for idx, t in max_times.items():
            if t > max_time or (t == max_time and idx < best_index):
                max_time = t
                best_index = idx
        return best_index