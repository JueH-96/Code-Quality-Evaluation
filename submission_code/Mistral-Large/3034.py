from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Create a set to store all covered points
        covered_points = set()

        # Iterate through each car's coordinates
        for start, end in nums:
            # Add all points from start to end to the set
            for point in range(start, end + 1):
                covered_points.add(point)

        # The number of covered points is the size of the set
        return len(covered_points)