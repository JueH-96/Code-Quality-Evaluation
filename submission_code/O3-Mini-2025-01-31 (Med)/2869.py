from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # dp[i][0] is the maximum contiguous non-decreasing subarray ending at i 
        # if we choose nums1[i], and similarly for dp[i][1] with nums2[i]
        dp = [[0, 0] for _ in range(n)]
        
        # Base case: a subarray with one element always has length 1
        dp[0][0] = 1
        dp[0][1] = 1
        
        res = 1
        
        for i in range(1, n):
            # For each possibility at the current index, initialize with 1 (starting new subarray)
            for j in range(2):
                curr_val = nums1[i] if j == 0 else nums2[i]
                # Start new subarray at index i
                best = 1
                # Try to extend the subarray from index i-1 choosing either option
                for k in range(2):
                    prev_val = nums1[i-1] if k == 0 else nums2[i-1]
                    if prev_val <= curr_val:
                        best = max(best, dp[i-1][k] + 1)
                dp[i][j] = best
                res = max(res, best)
                
        return res