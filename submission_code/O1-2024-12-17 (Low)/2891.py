from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # We'll treat each element nums[i] as defining an interval
        # [nums[i] - k, nums[i] + k] in which that index can be changed.
        # We want to find a single value x that lies in the maximum number
        # of these intervals (this corresponds to the longest subsequence
        # of equal elements we can create). So, our task reduces to
        # finding the maximum coverage of a point by these intervals.
        
        # We'll use a sweep line technique:
        #   - For each interval [L, R], we add two events:
        #       (L, +1)  meaning that at L we start covering
        #       (R+1, -1) meaning that after R we stop covering
        #
        #   - Then we sort these events by their coordinate and
        #     iterate through them, tracking how many intervals are
        #     currently covering. The maximum coverage at any point
        #     is the answer.
        
        events = []
        for val in nums:
            L = val - k
            R = val + k
            events.append((L, +1))
            events.append((R+1, -1))
        
        # Sort events by their coordinate
        events.sort(key=lambda x: x[0])
        
        coverage = 0
        max_coverage = 0
        for _, delta in events:
            coverage += delta
            max_coverage = max(max_coverage, coverage)
        
        return max_coverage