from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered_points = set()
        
        # nums is a list of [start, end] pairs representing car intervals
        for start_i, end_i in nums:
            # For each car, iterate through all integer points it covers
            # The range function's upper bound is exclusive, so end_i + 1 is needed
            for point in range(start_i, end_i + 1):
                covered_points.add(point)
                
        # The number of unique points covered is the size of the set
        return len(covered_points)