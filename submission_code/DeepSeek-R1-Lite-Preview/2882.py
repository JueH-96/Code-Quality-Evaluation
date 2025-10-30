class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        # Calculate the maximum y such that y^x <= n
        y_max = int(n ** (1/x))
        # Generate the list of y^x for y from 1 to y_max
        powers = [y**x for y in range(1, y_max + 1)]
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dfs(index, target):
            if target == 0:
                return 1
            if target < 0 or index >= len(powers):
                return 0
            # Include the current power
            count = dfs(index + 1, target - powers[index])
            # Exclude the current power
            count += dfs(index + 1, target)
            return count % MOD
        return dfs(0, n)