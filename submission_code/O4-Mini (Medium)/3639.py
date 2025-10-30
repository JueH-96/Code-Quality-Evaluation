from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # Build a difference array to count how many queries cover each index
        diff = [0] * (n + 1)
        for l, r in queries:
            diff[l] += 1
            # we can safely decrement diff[r+1] even when r == n-1,
            # because diff has length n+1 and we only read up to index n-1
            diff[r + 1] -= 1

        # Compute the cover count by prefix-summing diff
        cover_count = [0] * n
        curr = 0
        for i in range(n):
            curr += diff[i]
            cover_count[i] = curr

        # Check if each index can be decremented exactly nums[i] times
        for i in range(n):
            if cover_count[i] < nums[i]:
                return False
        return True