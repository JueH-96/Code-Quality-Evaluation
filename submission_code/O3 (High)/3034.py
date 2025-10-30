from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered_points = set()
        
        # Collect every integer point inside each interval [start, end]
        for start, end in nums:
            covered_points.update(range(start, end + 1))  # inclusive of `end`
        
        # The number of unique points covered by at least one car
        return len(covered_points)