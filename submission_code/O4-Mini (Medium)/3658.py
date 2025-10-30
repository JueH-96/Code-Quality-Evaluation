from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # M will track the maximum absolute diff between two known neighbors
        M = 0
        # A will collect all known values adjacent to a missing spot
        A = []
        n = len(nums)
        
        for i in range(n - 1):
            a, b = nums[i], nums[i + 1]
            # if both known, update M
            if a != -1 and b != -1:
                M = max(M, abs(a - b))
            # if exactly one is -1, record the known one
            elif a == -1 and b != -1:
                A.append(b)
            elif a != -1 and b == -1:
                A.append(a)
            # else both -1: nothing to do
        
        # If there are no known-to-missing adjacencies, we can make all numbers equal
        if not A:
            return M  # M might be 0 or from fully-known segments
        
        low, high = min(A), max(A)
        # best possible max-diff for those mixed edges is ceil((high-low)/2)
        t = (high - low + 1) // 2
        return max(M, t)