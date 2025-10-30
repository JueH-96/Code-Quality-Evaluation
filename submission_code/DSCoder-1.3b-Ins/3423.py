from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        prefix = [0]*(n+1)
        suffix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = (prefix[i] + nums[i]) % MOD
            suffix[i+1] = (suffix[i] + nums[n-i-1]) % MOD
        for i in range(n):
            suffix[i+1] = suffix[i+1] + suffix[i]
        ans = 0
        for pos, x in queries:
            ans = (ans + max(0, suffix[pos] - prefix[pos-1] + x)) % MOD
        return ans