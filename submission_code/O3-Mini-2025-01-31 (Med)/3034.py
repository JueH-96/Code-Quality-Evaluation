class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Create a boolean list to represent the points 0 to 100
        covered = [False] * 101
        # Mark each point covered by any car interval
        for start, end in nums:
            for point in range(start, end + 1):
                covered[point] = True
        # Sum up the True values to get the number of covered points
        return sum(covered)