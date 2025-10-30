class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def subtract_one(s):
            if s == "0":
                return "-1"
            digits = list(map(int, s))
            n = len(digits)
            borrow = 1
            for i in range(n-1, -1, -1):
                current = digits[i] - borrow
                if current < 0:
                    current += 10
                    borrow = 1
                else:
                    borrow = 0
                digits[i] = current
                if borrow == 0:
                    break
            i = 0
            while i < n and digits[i] == 0:
                i += 1
            if i == n:
                return "0"
            else:
                return ''.join(map(str, digits[i:]))
        
        def count(s):
            if not s:
                return 0
            digits = list(map(int, s))
            n = len(digits)
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, prev_digit, started, tight):
                if pos == n:
                    return 1 % MOD
                limit = digits[pos] if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    add = 0
                    if not started:
                        if d == 0:
                            add = dp(pos + 1, -1, False, new_tight)
                        else:
                            add = dp(pos + 1, d, True, new_tight)
                    else:
                        if abs(d - prev_digit) == 1:
                            add = dp(pos + 1, d, True, new_tight)
                    total = (total + add) % MOD
                return total
            
            return dp(0, -1, False, True) % MOD
        
        low_minus_1 = subtract_one(low)
        count_high = count(high)
        count_low = count(low_minus_1) if low_minus_1 != '-1' else 0
        return (count_high - count_low) % MOD