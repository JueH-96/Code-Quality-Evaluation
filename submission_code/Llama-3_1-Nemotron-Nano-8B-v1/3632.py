from typing import List

class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        totals = {}
        prev_time = None
        
        for index, time in events:
            if prev_time is None:
                totals[index] = time
            else:
                diff = time - prev_time
                totals[index] = totals.get(index, 0) + diff
            prev_time = time
        
        max_time = -1
        result = None
        for index in totals:
            current_time = totals[index]
            if current_time > max_time or (current_time == max_time and index < result):
                max_time = current_time
                result = index
        
        return result