from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Each number x defines an interval [x-k, x+k].
        intervals = [(x - k, x + k) for x in nums]
        # Sort intervals by their right endpoint (earliest finishing first).
        intervals.sort(key=lambda interval: interval[1])
        
        current = -10**20  # A very small number to start assigning from
        distinct_count = 0
        
        # Greedy: for each interval, assign the smallest integer >= current+1 and >= l
        # If that assignment is within the interval, we take it and count one more distinct.
        for l, r in intervals:
            assign = max(current + 1, l)
            if assign <= r:
                current = assign
                distinct_count += 1
        
        return distinct_count