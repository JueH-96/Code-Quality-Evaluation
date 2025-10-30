class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Create a set to store all unique points covered by any car
        covered_points = set()
        
        # Iterate through each car's start and end points
        for start, end in nums:
            # Add all points from start to end (inclusive) to the set
            for point in range(start, end + 1):
                covered_points.add(point)
        
        # Return the number of unique points covered
        return len(covered_points)