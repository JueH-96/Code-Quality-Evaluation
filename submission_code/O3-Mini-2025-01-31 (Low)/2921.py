MOD = 10**9 + 7

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        # Precompute counts for all lengths up to 100 separately,
        # for stepping numbers of exactly length L.
        # dp[l][d] = number of stepping numbers of length l ending with digit d.
        max_len = max(len(low), len(high))
        dp_table = [[0]*10 for _ in range(max_len+1)]
        # For length 1, any digit 0...9 is a valid number,
        # but when counting stepping numbers as a whole number, we do not allow 0 as the only digit.
        # However, for our recurrence we want to count all sequences including 0.
        for d in range(10):
            dp_table[1][d] = 1
        
        for l in range(2, max_len+1):
            for d in range(10):
                tot = 0
                if d - 1 >= 0:
                    tot += dp_table[l-1][d-1]
                if d + 1 <= 9:
                    tot += dp_table[l-1][d+1]
                dp_table[l][d] = tot % MOD

        # countAll(l) returns the count of stepping numbers with exactly l digits and no leading zero.
        def countAll(l: int) -> int:
            if l == 0:
                return 0
            # for 1-digit numbers we must exclude '0'
            if l == 1:
                return 9  # digits 1-9
            tot = 0
            # for multi-digit numbers, the first digit cannot be 0.
            for d in range(1, 10):
                tot = (tot + dp_table[l][d]) % MOD
            return tot

        # Calculate total count of stepping numbers with length strictly less than L.
        def count_smaller_lengths(L: int) -> int:
            tot = 0
            for l in range(1, L):
                tot = (tot + countAll(l)) % MOD
            return tot

        # The typical digit DP for numbers with the same number of digits as x.
        # We compute count_dp(x) = count of stepping numbers of length n (n = len(x))
        # that are <= x. Note that x doesn't have leading zeros.
        def count_dp(x: str) -> int:
            n = len(x)
            # use memoization: (pos, prev_digit, tight) 
            # pos: current position (0-indexed) in the number we are building.
            # prev_digit: the previous digit chosen (if pos==0, this doesn't exist and we pass -1)
            # tight: whether we are bound by the prefix of x
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos: int, prev: int, tight: bool) -> int:
                if pos == n:
                    # Completed building one number
                    return 1
                
                limit = int(x[pos]) if tight else 9
                total = 0
                if pos == 0:
                    # For the first digit, we cannot choose 0 (leading zero not allowed)
                    for d in range(1, limit+1):
                        # next state; newtight is True if d equals the digit in x; else False.
                        ntight = tight and (d == limit)
                        total = (total + dp(pos+1, d, ntight)) % MOD
                else:
                    # For subsequent positions, we must follow the stepping number rule:
                    # Only allowed digits are prev-1 and prev+1 (if in range).
                    for nd in (prev - 1, prev + 1):
                        if 0 <= nd <= limit and nd <= 9:
                            if nd > limit:
                                continue
                        if nd < 0 or nd > 9:
                            continue
                        if nd > limit and tight:
                            continue
                        ntight = tight and (nd == limit)
                        if nd <= limit:
                            total = (total + dp(pos+1, nd, ntight)) % MOD
                return total
            
            return dp(0, -1, True)
        
        # We want to count stepping numbers in range [low, high].
        # We'll create a helper function count(x) = # stepping numbers <= x.
        # count(x) = (all stepping numbers with length < len(x)) + (stepping numbers of length len(x) <= x)
        def count(x: str) -> int:
            L = len(x)
            res = count_smaller_lengths(L)
            res = (res + count_dp(x)) % MOD
            return res

        # A helper to decrement a number (in string form) by one.
        def dec_str(num: str) -> str:
            # Convert to list for easy manipulation.
            num_list = list(num)
            i = len(num_list) - 1
            while i >= 0:
                if num_list[i] != '0':
                    num_list[i] = str(int(num_list[i]) - 1)
                    break
                else:
                    num_list[i] = '9'
                    i -= 1
            # Remove leading zeros, but ensure at least one digit.
            res = "".join(num_list).lstrip('0')
            return res if res != "" else "0"
        
        # count stepping numbers in [low, high] = count(high) - count(low-1)
        low_dec = dec_str(low)
        res = (count(high) - count(low_dec)) % MOD
        return res