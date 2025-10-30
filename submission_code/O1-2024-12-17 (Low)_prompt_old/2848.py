class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        import sys
        sys.setrecursionlimit(10**7)

        MOD = 10**9 + 7
        n = len(nums)

        # Build adjacency: which index i can be directly followed by index j
        # in a special permutation (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0).
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        adj[i].append(j)

        # dp[mask][i] = number of ways to form a valid special arrangement
        # using the indices in "mask", ending with index i.
        dp = [[-1]*n for _ in range(1<<n)]

        def dfs(mask, last):
            # If we have used this state before, return memoized result.
            if dp[mask][last] != -1:
                return dp[mask][last]

            # If mask indicates we only chose 'last' in the subset:
            # There's exactly one way to have arrived here.
            if mask == (1 << last):
                dp[mask][last] = 1
                return 1

            ways = 0
            # Remove 'last' from mask to see the previous state
            prev_mask = mask ^ (1 << last)

            # Try all possible previous indices 'p' that could lead to 'last'
            # and were included in prev_mask.
            for p in adj[last]:
                if (prev_mask & (1 << p)) != 0:
                    ways = (ways + dfs(prev_mask, p)) % MOD

            dp[mask][last] = ways
            return ways

        full_mask = (1 << n) - 1
        ans = 0
        # Sum over all possible endings.
        for i in range(n):
            ans = (ans + dfs(full_mask, i)) % MOD

        return ans