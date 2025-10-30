class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        max_len = max(len(low), len(high))
        dp = [[0] * 10 for _ in range(max_len + 1)]
        for d in range(1, 10):
            dp[1][d] = 1
        for length in range(2, max_len + 1):
            for d in range(10):
                if d == 0:
                    dp[length][d] = dp[length - 1][1]
                elif d == 9:
                    dp[length][d] = dp[length - 1][8]
                else:
                    dp[length][d] = (dp[length - 1][d - 1] + dp[length - 1][d + 1]) % MOD
                    
        memo = {}
        
        def solve(index, is_tight, last_digit, s):
            if index == len(s):
                return 1
            if (index, is_tight, last_digit) in memo:
                return memo[(index, is_tight, last_digit)]
            count = 0
            limit = int(s[index]) if is_tight else 9
            for digit in range(limit + 1):
                if index == 0 and digit == 0:
                    continue
                if index > 0 and abs(digit - last_digit) != 1:
                    continue
                next_is_tight = is_tight and (digit == limit)
                count = (count + solve(index + 1, next_is_tight, digit, s)) % MOD
            memo[(index, is_tight, last_digit)] = count
            return count
            
        def count_stepping_upto_string(s):
            n = len(s)
            total_count = 0
            for length in range(1, n):
                count_l = sum(dp[length][d] for d in range(1, 10)) % MOD
                total_count = (total_count + count_l) % MOD
            memo.clear()
            count_n = solve(0, True, -1, s)
            total_count = (total_count + count_n) % MOD
            return total_count
            
        high_count = count_stepping_upto_string(high)
        low_int = int(low)
        if low_int > 1:
            low_minus_1_str = str(low_int - 1)
            low_minus_1_count = count_stepping_upto_string(low_minus_1_str)
        else:
            low_minus_1_count = 0
            
        result = (high_count - low_minus_1_count + MOD) % MOD
        return result