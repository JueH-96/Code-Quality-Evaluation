class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute the last occurrence of each number
        last_occurrence = {}
        for i in range(n):
            last_occurrence[nums[i]] = i
        
        # Initialize DP and prefix sum arrays
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty partition
        prefix = [0] * (n + 1)
        prefix[0] = 1
        
        furthest = 0
        for i in range(1, n + 1):
            current = nums[i - 1]
            # Update the furthest position that any element in the current window can reach
            furthest = max(furthest, last_occurrence[current])
            
            # If current index reaches the furthest, we can make a split here
            if i == furthest:
                # The number of ways to split up to this point is the sum of all ways up to the start of the current window
                # This is efficiently computed using the prefix sum array
                dp[i] = (prefix[i - 1] - (prefix[furthest - 1] if furthest > 0 else 0)) % MOD
                prefix[i] = (prefix[i - 1] + dp[i]) % MOD
            else:
                # If we haven't reached the furthest, no new splits are possible here
                prefix[i] = prefix[i - 1]
        
        return dp[n] % MOD