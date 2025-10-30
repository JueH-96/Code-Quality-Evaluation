class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        import sys
        MOD = 10**9 + 7
        
        n = len(nums)
        # Edge case: if there's only one element, we can pick arr1[0] = 0..nums[0]
        # and arr2[0] = nums[0] - arr1[0], yielding (nums[0] + 1) ways
        if n == 1:
            return (nums[0] + 1) % MOD
        
        # dp[i][v] = number of ways such that x[i] = v, where x[i] = arr1[i]
        # and 0 <= x[i] <= nums[i].
        # We'll only store up to 50 because nums[i] <= 50.
        dp = [[0]*51 for _ in range(n)]
        
        # Initialize dp for i = 0
        for v in range(nums[0] + 1):
            dp[0][v] = 1
        
        # Build prefix sums to make transitions faster
        def build_prefix(arr):
            prefix = [0]*51
            prefix[0] = arr[0] % MOD
            for i in range(1, 51):
                prefix[i] = (prefix[i-1] + arr[i]) % MOD
            return prefix
        
        # Fill dp for i = 1..n-1 using the constraints
        for i in range(n - 1):
            prefix_dp = build_prefix(dp[i])
            
            # The difference needed: x[i+1] >= x[i] + max(0, nums[i+1] - nums[i])
            d = max(0, nums[i+1] - nums[i])
            
            # Compute dp[i+1][:]
            for w in range(nums[i+1] + 1):
                if w >= d:
                    dp[i+1][w] = prefix_dp[w - d]  # sum of dp[i][0..(w-d)]
                else:
                    dp[i+1][w] = 0
        
        # The answer is the sum of dp[n-1][0..nums[n-1]]
        ans = sum(dp[n-1][v] for v in range(nums[n-1] + 1)) % MOD
        return ans