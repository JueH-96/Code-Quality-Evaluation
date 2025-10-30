class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = 10**9 + 7
        # Precompute unique numbers raised to the power x that are <= n.
        # Since n is small, we can iterate through possible base numbers.
        nums = []
        base = 1
        while True:
            power_val = base ** x
            if power_val > n:
                break
            nums.append(power_val)
            base += 1
        
        # Use recursion with memoization to count ways.
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(i, remaining):
            # if we have exactly reached 0, we count it as one valid way.
            if remaining == 0:
                return 1
            # if remaining becomes negative or we've exhausted the list, no valid way.
            if remaining < 0 or i == len(nums):
                return 0
            
            # Option1: include current number and move to next index.
            take = dfs(i + 1, remaining - nums[i])
            # Option2: skip current number.
            not_take = dfs(i + 1, remaining)
            return (take + not_take) % mod
        
        return dfs(0, n)