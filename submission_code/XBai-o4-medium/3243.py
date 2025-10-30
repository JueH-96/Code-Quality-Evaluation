class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_val = int(s)
        len_s = len(s)
        finish_str = str(finish)
        max_len_finish = len(finish_str)
        
        if len_s > max_len_finish:
            return 0
        
        total = 0
        
        # Handle k=0 case
        if len_s <= max_len_finish:
            if s_val >= start and s_val <= finish:
                total += 1
        
        def count_valid(upper_str: str, k: int, limit: int) -> int:
            if len(upper_str) < k:
                return 0
            if len(upper_str) > k:
                return limit * ((limit + 1) ** (k - 1))
            digits = [int(c) for c in upper_str]
            n = k
            dp = [[0] * 2 for _ in range(n + 1)]
            dp[0][1] = 1  # Starting with tight=True at position 0
            for i in range(n):
                for tight in [0, 1]:
                    current = dp[i][tight]
                    if current == 0:
                        continue
                    max_d = digits[i] if tight else 9
                    for d in range(0, max_d + 1):
                        if i == 0 and d == 0:
                            continue  # Skip leading zero
                        if d > limit:
                            continue
                        new_tight = 0
                        if tight:
                            if d == max_d:
                                new_tight = 1
                        dp[i+1][new_tight] += current
            return dp[n][0] + dp[n][1]
        
        possible_k_max = max_len_finish - len_s
        for k in range(1, possible_k_max + 1):
            m = k + len_s
            min_m = 10 ** (m - 1)
            max_m = (10 ** m) - 1
            if max_m < start or min_m > finish:
                continue
            x_low = max(start, min_m)
            x_high = min(finish, max_m)
            if x_high < x_low:
                continue
            C = 10 ** len_s
            S_val = s_val
            numerator_low = x_low - S_val
            if numerator_low <= 0:
                P_min_num = 10 ** (k - 1)
            else:
                P_min_num = (numerator_low + C - 1) // C  # ceil division
            P_min_num = max(P_min_num, 10 ** (k - 1))
            numerator_high = x_high - S_val
            P_max_num = numerator_high // C
            P_max_num = min(P_max_num, 10 ** k - 1)
            if P_min_num > P_max_num:
                continue
            upper_str = str(P_max_num)
            lower_str = str(P_min_num - 1)
            count_upper_pmax = count_valid(upper_str, k, limit)
            count_upper_pmin_minus_1 = count_valid(lower_str, k, limit)
            total += count_upper_pmax - count_upper_pmin_minus_1
        
        return total