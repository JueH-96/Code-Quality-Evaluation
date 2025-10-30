from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        points = set()
        for start, end in nums:
            for point in range(start, end + 1):
                points.add(point)
        return len(points)