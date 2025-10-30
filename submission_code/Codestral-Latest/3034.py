class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Create a set to store all the points covered by the cars
        covered_points = set()

        # Iterate through each car's parking interval
        for start, end in nums:
            # Add all points in the interval [start, end] to the set
            for point in range(start, end + 1):
                covered_points.add(point)

        # The number of unique points covered is the size of the set
        return len(covered_points)