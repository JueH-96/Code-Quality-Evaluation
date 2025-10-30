class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize the possible ranges for arr1 and arr2
        # arr1 is non-decreasing, arr2 is non-increasing
        # arr1[i] + arr2[i] = nums[i]
        # So, arr1[i] can be from 0 to nums[i], and arr2[i] = nums[i] - arr1[i]
        
        # To ensure arr1 is non-decreasing and arr2 is non-increasing:
        # arr1[i] >= arr1[i-1]
        # arr2[i] <= arr2[i-1]
        # Which translates to:
        # nums[i] - arr1[i] <= nums[i-1] - arr1[i-1]
        # => arr1[i] >= arr1[i-1] + (nums[i] - nums[i-1])
        
        # So, for each i, arr1[i] must be >= arr1[i-1] and >= arr1[i-1] + (nums[i] - nums[i-1])
        # Similarly, arr2[i] must be <= arr2[i-1] and <= nums[i] - arr1[i]
        
        # We can use dynamic programming to count the number of valid sequences
        
        # Initialize the DP table
        # dp[i][a] represents the number of ways to assign arr1[0..i] such that arr1[i] = a
        # Since arr1 is non-decreasing, a can be from 0 to nums[i]
        
        # Initialize for the first element
        dp = [ [0] * (nums[0] + 1) for _ in range(n) ]
        for a in range(nums[0] + 1):
            dp[0][a] = 1
        
        for i in range(1, n):
            new_dp = [0] * (nums[i] + 1)
            for prev_a in range(nums[i-1] + 1):
                # arr1[i] must be >= prev_a and >= prev_a + (nums[i] - nums[i-1])
                min_a = max(prev_a, prev_a + (nums[i] - nums[i-1]))
                if min_a > nums[i]:
                    continue
                for a in range(min_a, nums[i] + 1):
                    new_dp[a] = (new_dp[a] + dp[i-1][prev_a]) % MOD
            dp[i] = new_dp
        
        # The total number of ways is the sum of all possible arr1[n-1]
        total = sum(dp[n-1]) % MOD
        return total