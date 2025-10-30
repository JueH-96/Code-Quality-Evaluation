from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # We will use a difference array to count how many queries cover each index.
        diff = [0] * (n + 1)
        
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n:
                diff[r + 1] -= 1
                
        # Convert the difference array to the actual coverage counts.
        current = 0
        for i in range(n):
            current += diff[i]
            # For each index, if coverage is less than required decrements (nums[i]),
            # then it's impossible to bring that index to zero.
            if current < nums[i]:
                return False
        
        return True