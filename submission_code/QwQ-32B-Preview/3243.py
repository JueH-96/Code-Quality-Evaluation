class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        m = len(s)
        return self.count_powerful_up_to(finish, m, s, limit) - self.count_powerful_up_to(start - 1, m, s, limit)
    
    def count_powerful_up_to(self, n, m, s, limit):
        sn = str(n)
        k = len(sn)
        
        if k < m:
            return 0
        
        total = 0
        
        # Part 1: p from 0 to k - m - 1
        if m <= k:
            # p = 0
            if m == k or int(sn[:m]) >= int(s):
                total += 1
            for p in range(1, k - m):
                # Number of p-digit y with each digit <= limit
                if p == 1:
                    total += limit
                else:
                    total += limit * (limit + 1)**(p - 1)
        
        # Part 2: p = k - m
        if k - m >= 0:
            @lru_cache(None)
            def dp(pos, tight, non_zero):
                if pos == k:
                    return 1 if non_zero or (k - m == 0) else 0
                if pos < k - m:
                    max_d = min(limit, int(sn[pos])) if tight else limit
                    res = 0
                    for d in range(0, max_d + 1):
                        new_tight = tight and (d == int(sn[pos]))
                        new_non_zero = non_zero or (d > 0)
                        res += dp(pos + 1, new_tight, new_non_zero)
                    return res
                else:
                    required_d = int(s[pos - (k - m)])
                    if tight and required_d > int(sn[pos]):
                        return 0
                    new_tight = tight and (required_d == int(sn[pos]))
                    new_non_zero = non_zero or (required_d > 0)
                    return dp(pos + 1, new_tight, new_non_zero)
            
            total += dp(0, True, False)
        
        return total