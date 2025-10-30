from functools import lru_cache

def waysToReachStair(k):
    MOD = 10**9 + 7

    @lru_cache(maxsize=None)
    def dfs(current, can_down, num_up_steps):
        if current == k:
            return 1
        if current > k:
            if can_down:
                if current - 1 == k:
                    return 1
                else:
                    return 0
            else:
                new_current = current + (2 ** num_up_steps)
                return dfs(new_current, True, num_up_steps + 1)
        else:
            res = 0
            if can_down:
                if current - 1 >= 0:
                    res += dfs(current - 1, False, num_up_steps)
                jump = 2 ** num_up_steps
                new_current = current + jump
                res += dfs(new_current, True, num_up_steps + 1)
                res %= MOD
            else:
                jump = 2 ** num_up_steps
                new_current = current + jump
                res += dfs(new_current, True, num_up_steps + 1)
                res %= MOD
            return res % MOD

    return dfs(1, False, 0) % MOD