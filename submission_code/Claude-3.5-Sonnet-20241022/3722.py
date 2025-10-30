class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Calculate prefix sums for efficient subarray sum computation
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
            
        # dp[i][j] represents the maximum sum possible using j subarrays
        # considering elements from index i onwards
        dp = {}
        
        def solve(start: int, remaining: int) -> int:
            if remaining == 0:
                return 0
            if start >= n:
                return float('-inf')
            
            state = (start, remaining)
            if state in dp:
                return dp[state]
            
            # Don't include current position in any subarray
            result = solve(start + 1, remaining)
            
            # Try all possible subarray lengths starting from m
            for length in range(m, n - start + 1):
                subarray_sum = prefix[start + length] - prefix[start]
                next_sum = solve(start + length, remaining - 1)
                if next_sum != float('-inf'):
                    result = max(result, subarray_sum + next_sum)
            
            dp[state] = result
            return result
        
        return solve(0, k)