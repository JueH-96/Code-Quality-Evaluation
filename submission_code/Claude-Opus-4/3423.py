class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        total_sum = 0
        
        def maxNonAdjacentSum(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return max(0, arr[0])
            
            # dp[i] represents max sum considering elements from 0 to i
            prev2 = max(0, arr[0])  # dp[i-2]
            prev1 = max(prev2, arr[1])  # dp[i-1]
            
            for i in range(2, len(arr)):
                current = max(prev1, arr[i] + prev2)
                prev2 = prev1
                prev1 = current
            
            return prev1
        
        # Process each query
        for pos, x in queries:
            nums[pos] = x
            max_sum = maxNonAdjacentSum(nums)
            total_sum = (total_sum + max_sum) % MOD
        
        return total_sum