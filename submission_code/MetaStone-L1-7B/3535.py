MOD = 10**9 + 7

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize DP for the first element
        dp_prev = [0] * (n + 1)
        dp_prev[0] = 1  # Base case: one way to have a_prev = 0 at i=0
        
        for i in range(1, n):
            dp_current = [0] * (n + 1)
            for a_prev in range(n + 1):
                if dp_prev[a_prev] == 0:
                    continue
                # Calculate the minimum a_curr based on the condition
                lower = a_prev + max(0, nums[i-1] - nums[i])
                # a_curr can be from lower to min(nums[i], n)
                max_a_curr = min(nums[i], n)
                for a_curr in range(lower, max_a_curr + 1):
                    dp_current[a_curr] = (dp_current[a_curr] + dp_prev[a_prev]) % MOD
            dp_prev = dp_current
        
        return sum(dp_prev) % MOD