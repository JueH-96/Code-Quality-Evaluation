from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Since 1 <= start_i <= end_i <= 100, we can use a fixed-size array to mark covered points.
        covered = [False] * 101  # indices 0 through 100

        for start, end in nums:
            for point in range(start, end + 1):
                covered[point] = True

        # Count how many points are covered
        return sum(covered)