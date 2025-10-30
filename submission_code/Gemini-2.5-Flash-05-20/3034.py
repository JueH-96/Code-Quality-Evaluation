from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Initialize a boolean array to keep track of covered integer points.
        # The coordinates are between 1 and 100, so an array of size 101
        # allows us to use indices 1 through 100 directly.
        # covered_points[i] will be True if integer point i is covered by any car.
        covered_points = [False] * 101 

        # Iterate through each car's coordinates [start, end]
        for start, end in nums:
            # For each car, mark all integer points from its start to its end (inclusive)
            # as covered.
            # The range function's second argument is exclusive, so we use end + 1.
            for point in range(start, end + 1):
                covered_points[point] = True
        
        # Count the total number of unique integer points that are covered.
        # We iterate from index 1 to 100, as these are the possible coordinates.
        count = 0
        for i in range(1, 101):
            if covered_points[i]:
                count += 1
        
        return count