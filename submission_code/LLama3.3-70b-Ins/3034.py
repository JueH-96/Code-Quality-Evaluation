from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Create a set to store unique points covered by cars
        points = set()
        
        # Iterate over each car
        for start, end in nums:
            # Add all points covered by the current car to the set
            for point in range(start, end + 1):
                points.add(point)
        
        # Return the total number of unique points covered by cars
        return len(points)