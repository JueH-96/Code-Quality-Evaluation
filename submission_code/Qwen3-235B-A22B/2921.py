class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        # Precompute adjacent digits
        adjacent = [[] for _ in range(10)]
        for d in range(10):
            if d > 0:
                adjacent[d].append(d - 1)
            if d < 9:
                adjacent[d].append(d + 1)
        
        # Precompute pre_dp[k][d]: number of k-digit stepping numbers ending with digit d
        MAX_K = 100
        pre_dp = [ [0] * 10 for _ in range(MAX_K + 1) ]
        for d in range(10):
            pre_dp[1][d] = 1 if d >= 1 else 0
        
        for k in range(2, MAX_K + 1):
            for d in range(10):
                for m in adjacent[d]:
                    pre_dp[k][d] += pre_dp[k-1][m]
                    pre_dp[k][d] %= MOD
        
        # Helper function to subtract 1 from a string number
        def subtract_one(s):
            digits = list(s)
            i = len(digits) - 1
            while i >= 0 and digits[i] == '0':
                digits[i] = '9'
                i -= 1
            if i == -1:
                return '0'  # should not happen per problem constraints
            digits[i] = str(int(digits[i]) - 1)
            # Remove leading zeros
            if digits[0] == '0' and len(digits) > 1:
                return ''.join(digits[1:])
            else:
                return ''.join(digits)
        
        # Function to compute part1: all stepping numbers with fewer digits than s
        def get_part1(s):
            len_s = len(s)
            if len_s <= 0:
                return 0
            total = 0
            for k in range(1, len_s):
                current = 0
                for d in range(10):
                    current = (current + pre_dp[k][d]) % MOD
                total = (total + current) % MOD
            return total
        
        # Function to compute part2: stepping numbers with exactly len(s) digits and <= s
        def get_part2(s):
            n = len(s)
            if n == 0:
                return 0
            digits = [int(c) for c in s]
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, prev_digit, tight):
                if pos == n:
                    return 1
                res = 0
                upper = digits[pos] if tight else 9
                for d in range(0, upper + 1):
                    if pos == 0:
                        if d == 0:
                            continue  # first digit must be non-zero
                        new_tight = tight and (d == upper)
                        res = (res + dp(pos + 1, d, new_tight)) % MOD
                    else:
                        if abs(d - prev_digit) != 1:
                            continue
                        new_tight = tight and (d == upper)
                        res = (res + dp(pos + 1, d, new_tight)) % MOD
                return res % MOD
            
            return dp(0, -1, True)
        
        # Total count function
        def count(s):
            if s == "0":
                return 0
            part1 = get_part1(s)
            part2 = get_part2(s)
            total = (part1 + part2) % MOD
            return total
        
        # Compute low-1 and get counts
        low_minus_1 = subtract_one(low)
        count_high = count(high)
        count_low_minus1 = count(low_minus_1)
        ans = (count_high - count_low_minus1) % MOD
        # Handle negative values
        return ans if ans >=0 else ans + MOD