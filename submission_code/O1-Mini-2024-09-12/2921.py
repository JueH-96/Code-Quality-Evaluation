class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def decrement(num: str) -> str:
            num = list(num)
            i = len(num) - 1
            while i >= 0 and num[i] == '0':
                num[i] = '9'
                i -= 1
            if i == -1:
                return '0'
            num[i] = str(int(num[i]) - 1)
            # Remove leading zeros
            return ''.join(num).lstrip('0') or '0'
        
        from functools import lru_cache
        
        def count(num: str) -> int:
            n = len(num)
            
            @lru_cache(None)
            def dp(i: int, tight: bool, started: bool, prev: int) -> int:
                if i == n:
                    return int(started)
                limit = int(num[i]) if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    new_started = started or (d != 0)
                    if not new_started:
                        total += dp(i + 1, new_tight, new_started, -1)
                    else:
                        if prev == -1 or abs(d - prev) == 1:
                            total += dp(i + 1, new_tight, new_started, d)
                return total % MOD
            
            return dp(0, True, False, -1)
        
        low_minus_1 = decrement(low)
        count_high = count(high)
        count_low = count(low_minus_1) if low_minus_1 != '0' else 0
        return (count_high - count_low) % MOD