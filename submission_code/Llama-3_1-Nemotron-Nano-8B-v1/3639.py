from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        delta = [0] * (n + 1)
        
        # Process each query to update the delta array
        for l, r in queries:
            delta[l] += 1
            if r + 1 < n:
                delta[r + 1] -= 1
        
        # Compute the count of how many queries cover each index
        c = []
        current = 0
        for i in range(n):
            current += delta[i]
            c.append(current)
        
        # Check if each index's coverage is at least its value
        for i in range(n):
            if c[i] < nums[i]:
                return False
        
        # Calculate the sum of maximum possible decrements (sum_max)
        sum_max = sum((r - l + 1) for l, r in queries)
        
        # Check if the sum of nums is within the possible decrements
        sum_nums = sum(nums)
        if sum_nums > sum_max:
            return False
        
        return True