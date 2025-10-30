class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def maxNonAdjacent(arr):
            n = len(arr)
            if n == 0:
                return 0
            if n == 1:
                return max(0, arr[0])
            
            # DP approach with space optimization
            # prev2 = dp[i-2], prev1 = dp[i-1]
            prev2 = max(0, arr[0])  # Can choose empty subsequence if negative
            prev1 = max(prev2, arr[1])
            
            for i in range(2, n):
                curr = max(prev1, prev2 + arr[i])
                prev2 = prev1
                prev1 = curr
            
            return prev1
        
        total = 0
        for pos, x in queries:
            nums[pos] = x
            result = maxNonAdjacent(nums)
            total = (total + result) % MOD
        
        return total