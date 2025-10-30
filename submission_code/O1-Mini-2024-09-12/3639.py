from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        count = [0] * (n + 1)
        
        for l, r in queries:
            count[l] += 1
            if r + 1 < n:
                count[r + 1] -= 1
        
        # Compute the prefix sum to get the number of queries covering each index
        for i in range(1, n):
            count[i] += count[i - 1]
        
        # Check if each count is at least the corresponding nums value
        for i in range(n):
            if count[i] < nums[i]:
                return False
        
        return True