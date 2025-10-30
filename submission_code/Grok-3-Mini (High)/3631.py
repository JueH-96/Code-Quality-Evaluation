import math
from functools import lru_cache

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 1000000007
        L = len(s)
        max_y = 800
        # Precompute ops_val
        ops_val = [0] * (max_y + 1)
        ops_val[1] = 0
        for y in range(2, max_y + 1):
            pop_y = bin(y).count('1')
            ops_val[y] = 1 + ops_val[pop_y]
        # Compute is_good
        is_good = [False] * (max_y + 1)
        for p in range(1, max_y + 1):
            if p == 1 or (p > 1 and ops_val[p] <= k - 1):
                is_good[p] = True
        # Convert s to nums
        nums = [int(c) for c in s]
        # Define dfs with memoization
        @lru_cache(None)
        def dfs(pos: int, cnt: int, tight: int) -> int:
            if pos == L:
                return 1 if is_good[cnt] else 0
            ans = 0
            for d in range(2):  # 0 and 1
                if not tight or d <= nums[pos]:
                    new_tight = 1 if (tight and d == nums[pos]) else 0
                    new_cnt = cnt + d
                    ans += dfs(pos + 1, new_cnt, new_tight)
                    ans %= MOD
            return ans % MOD
        # Call dfs
        result = dfs(0, 0, 1)
        # Compute pop_n
        pop_n = sum(nums)
        # Compute answer with modulo
        ans = (result - (1 if is_good[pop_n] else 0) + MOD) % MOD
        return ans