class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def maxNonAdjSum(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return max(0, arr[0])
                
            # dp[i] represents max sum up to index i
            dp = [0] * len(arr)
            dp[0] = max(0, arr[0])
            dp[1] = max(0, arr[0], arr[1])
            
            for i in range(2, len(arr)):
                # Either take current element + dp[i-2] or skip current element
                dp[i] = max(dp[i-1], dp[i-2] + arr[i], arr[i])
            
            return dp[-1]
            
        total = 0
        nums_copy = nums.copy()
        
        for pos, x in queries:
            nums_copy[pos] = x
            total = (total + maxNonAdjSum(nums_copy)) % MOD
            
        return total