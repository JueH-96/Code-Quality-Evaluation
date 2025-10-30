from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Create a boolean array (list) to keep track of covered integer points on the number line.
        # The coordinates are between 1 and 100. We need to cover this range.
        # Using a list of size 101, index i will correspond to point i.
        # Initialize all points from 1 to 100 as not covered.
        # The 0th index is unused for points in the range [1, 100].
        covered_points = [False] * 101 # covered_points[i] will be True if point i is covered

        # Iterate through each car interval [start_i, end_i] in the input list 'nums'.
        for start, end in nums:
            # For each interval, mark all integer points from start_i to end_i (inclusive)
            # as covered.
            # The range function `range(start, end + 1)` generates integers from `start` up to `end`.
            for point in range(start, end + 1):
                # Based on the constraints (1 <= start <= end <= 100),
                # the 'point' value will always be within the valid index range [1, 100] for our list.
                covered_points[point] = True

        # Count the total number of distinct integer points that are marked as covered.
        count = 0
        # Iterate through the relevant points, which are from 1 to 100 based on the problem constraints.
        for point in range(1, 101): # This loop checks points 1, 2, ..., 100
            # If the point was marked as covered by any car, increment the count.
            if covered_points[point]:
                count += 1

        return count