import math

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 1000000007
        
        def count_up_to(num_str: str) -> int:
            L = len(num_str)
            # Compute dp_len for stepping numbers of exact length
            dp_len_list = [None] * 101
            dp_len_list[1] = [0] * 10
            for dig in range(10):
                if 1 <= dig <= 9:
                    dp_len_list[1][dig] = 1
                else:
                    dp_len_list[1][dig] = 0
            for len_ in range(2, 101):
                dp_len_list[len_] = [0] * 10
                for dig in range(10):
                    if 0 <= dig - 1 <= 9:
                        dp_len_list[len_][dig] += dp_len_list[len_ - 1][dig - 1]
                        dp_len_list[len_][dig] %= MOD
                    if 0 <= dig + 1 <= 9:
                        dp_len_list[len_][dig] += dp_len_list[len_ - 1][dig + 1]
                        dp_len_list[len_][dig] %= MOD
            
            # Compute total per length
            total_per_len = [0] * 101
            for len_ in range(1, 101):
                total = 0
                for dig in range(10):
                    total += dp_len_list[len_][dig]
                    total %= MOD
                total_per_len[len_] = total
            
            # Compute cumulative sum
            cum_sum = [0] * 101
            for i in range(1, 101):
                cum_sum[i] = (cum_sum[i - 1] + total_per_len[i]) % MOD
            
            # Sum for lengths less than L
            sum_less = cum_sum[L - 1]
            
            # Now for exact length L with <= num_str using digit DP
            memo = [[[-1 for _ in range(2)] for _ in range(11)] for _ in range(L)]
            
            def dp(pos, prev, tight):
                if pos == L:
                    return 1
                if memo[pos][prev][tight] != -1:
                    return memo[pos][prev][tight]
                ans = 0
                if prev == 10:  # No previous digit, first digit
                    min_dig = 1
                    upper_dig = 9 if not tight else int(num_str[pos])
                    for d in range(min_dig, upper_dig + 1):
                        new_tight_val = 1 if tight and d == int(num_str[pos]) else 0
                        ans += dp(pos + 1, d, new_tight_val)
                        ans %= MOD
                else:  # Have previous digit
                    for delta in [-1, 1]:
                        cand_d = prev + delta
                        if 0 <= cand_d <= 9:
                            if tight:
                                if cand_d <= int(num_str[pos]):
                                    new_tight_val = 1 if cand_d == int(num_str[pos]) else 0
                                    ans += dp(pos + 1, cand_d, new_tight_val)
                                    ans %= MOD
                            else:  # Not tight
                                new_tight_val = 0
                                ans += dp(pos + 1, cand_d, new_tight_val)
                                ans %= MOD
                ans %= MOD
                memo[pos][prev][tight] = ans
                return ans
            
            count_exact = dp(0, 10, 1)
            res = (sum_less + count_exact) % MOD
            return res
        
        count_high = count_up_to(high)
        low_num = int(low)
        dec_low_str = str(low_num - 1)
        count_low_minus_one = count_up_to(dec_low_str)
        result = (count_high - count_low_minus_one + MOD) % MOD
        return result