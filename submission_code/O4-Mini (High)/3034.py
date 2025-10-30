from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Since 1 <= start_i <= end_i <= 100, we can use a fixed-size boolean array
        covered = [False] * 101
        
        # Mark every integer point covered by any car interval
        for start, end in nums:
            for point in range(start, end + 1):
                covered[point] = True
        
        # Count how many points are marked True
        return sum(covered)