MOD = 10**9 + 7

def subtract_one(s):
    s_list = list(s)
    i = len(s_list) - 1
    while i >= 0 and s_list[i] == '0':
        s_list[i] = '9'
        i -= 1
    if i < 0:
        return "0"
    s_list[i] = str(int(s_list[i]) - 1)
    new_s = ''.join(s_list).lstrip('0')
    if not new_s:
        return "0"
    return new_s

# Precompute the number of stepping numbers for each length from 1 to 100
precompute = {}
for l in range(1, 101):
    if l == 1:
        precompute[l] = 9
        continue
    total = 0
    for start in range(1, 10):
        dp_prev = [0] * 10
        dp_prev[start] = 1
        for _ in range(l - 1):
            dp_next = [0] * 10
            for d in range(10):
                if dp_prev[d] == 0:
                    continue
                if d + 1 <= 9:
                    dp_next[d + 1] += dp_prev[d]
                if d - 1 >= 0:
                    dp_next[d - 1] += dp_prev[d]
            dp_prev = dp_next
        total += sum(dp_prev)
    precompute[l] = total

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        def count(s):
            if s == "0":
                return 0
            n = len(s)
            total = 0
            # Add counts for lengths less than n
            for l in range(1, n):
                if l in precompute:
                    total += precompute[l]
            # Compute same length count
            digits = list(map(int, s))
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, prev, tight):
                if pos == n:
                    return 1
                limit = digits[pos] if tight else 9
                total = 0
                if pos == 0:
                    for d in range(1, 10):
                        if d > limit:
                            continue
                        new_tight = tight and (d == limit)
                        total += dp(pos + 1, d, new_tight)
                else:
                    for delta in (-1, 1):
                        d = prev + delta
                        if 0 <= d <= 9:
                            if d > limit:
                                continue
                            new_tight = tight and (d == limit)
                            total += dp(pos + 1, d, new_tight)
                return total

            same_length = dp(0, -1, True)
            total += same_length
            return total

        low_minus_1 = subtract_one(low)
        count_high = count(high)
        count_low = count(low_minus_1)
        result = (count_high - count_low) % MOD
        return result