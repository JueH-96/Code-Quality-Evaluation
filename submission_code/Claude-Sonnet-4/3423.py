class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def maxNonAdjacent(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return max(0, arr[0])
            
            # dp[i] represents max sum considering elements 0 to i
            prev2 = max(0, arr[0])  # dp[0]
            prev1 = max(prev2, arr[1])  # dp[1]
            
            for i in range(2, len(arr)):
                current = max(prev1, arr[i] + prev2)
                prev2 = prev1
                prev1 = current
            
            return prev1
        
        total_sum = 0
        
        for pos, x in queries:
            # Update the array
            nums[pos] = x
            
            # Calculate maximum sum of non-adjacent subsequence
            max_sum = maxNonAdjacent(nums)
            total_sum = (total_sum + max_sum) % MOD
        
        return total_sum