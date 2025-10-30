from functools import lru_cache

class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        return self.F(r) - self.F(l - 1)
    
    def F(self, n: int) -> int:
        if n == 0:
            return 0
        s = str(n)
        L = len(s)
        max_sum = min(90, 9 * L)
        total = 0
        for S in range(1, max_sum + 1):
            total += self.dp_count(s, S)
        return total
    
    def dp_count(self, s: str, S: int) -> int:
        L = len(s)
        
        @lru_cache(maxsize=None)
        def dfs(pos, tight, started, sum_val, rem):
            if pos == L:
                if started and sum_val == S and rem == 0:
                    return 1
                return 0
            total = 0
            current_limit = int(s[pos]) if tight else 9
            for d in range(0, current_limit + 1):
                new_tight = tight and (d == current_limit)
                if not started and d == 0:
                    total += dfs(pos + 1, new_tight, False, sum_val, rem)
                else:
                    new_started = True
                    new_sum = sum_val + d
                    if new_sum > S:
                        break
                    if not started:
                        new_rem = d % S
                    else:
                        new_rem = (rem * d) % S
                    total += dfs(pos + 1, new_tight, new_started, new_sum, new_rem)
            return total
        
        return dfs(0, True, False, 0, 1)