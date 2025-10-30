class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        m = len(s)
        
        # Precompute steps_dict for numbers up to 800
        max_steps = 800
        steps_dict = {1: 0}
        for x in range(2, max_steps + 1):
            bits_x = bin(x).count('1')
            steps_dict[x] = 1 + steps_dict.get(bits_x, 0)
        
        # Precompute comb table
        max_n = 800
        comb = [[0] * (max_n + 1) for _ in range(max_n + 1)]
        comb[0][0] = 1
        for n in range(1, max_n + 1):
            comb[n][0] = 1
            comb[n][n] = 1
            for k_comb in range(1, n):
                comb[n][k_comb] = (comb[n-1][k_comb-1] + comb[n-1][k_comb]) % MOD
        
        # Compute count_less_length: numbers with length 1 to m-1
        count_less_length = 0
        for l in range(1, m):
            if l == 1:
                if 0 <= k:
                    count_less_length += 1
                    count_less_length %= MOD
            else:
                for t in range(1, l + 1):
                    steps_x = 1 + steps_dict.get(t, 0)
                    if steps_x <= k:
                        count_less_length += comb[l-1][t-1]
                        count_less_length %= MOD
        
        # If m == 1, there are no numbers of equal length less than s
        if m == 1:
            return count_less_length % MOD
        
        # Compute count_equal_length_less: numbers of length m, less than s
        from collections import defaultdict
        dp = [defaultdict(int) for _ in range(m)]
        count_ones = 1
        tight = True
        dp[0][(count_ones, tight)] = 1
        
        for i in range(m - 1):
            current_dp = dp[i]
            for (current_count, current_tight), ways in current_dp.items():
                next_digit_pos = i + 1
                current_max_d = int(s[next_digit_pos]) if current_tight else 1
                for d in [0, 1]:
                    if d > current_max_d:
                        continue
                    new_tight = current_tight and (d == int(s[next_digit_pos]))
                    new_count = current_count + d
                    key = (new_count, new_tight)
                    dp[i+1][key] = (dp[i+1].get(key, 0) + ways) % MOD
        
        count_equal_length_less = 0
        for (count_ones, tight), ways in dp[m-1].items():
            if not tight:
                if count_ones == 1 and m == 1:
                    steps_x = 0
                else:
                    steps_x = 1 + steps_dict.get(count_ones, 0)
                if steps_x <= k:
                    count_equal_length_less += ways
                    count_equal_length_less %= MOD
        
        total = (count_less_length + count_equal_length_less) % MOD
        return total