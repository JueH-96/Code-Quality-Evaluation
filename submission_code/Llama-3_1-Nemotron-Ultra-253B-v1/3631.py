MOD = 10**9 + 7

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        max_n = 800
        # Precompute combinations C(n, k)
        comb = [[0] * (max_n + 1) for _ in range(max_n + 1)]
        for n in range(max_n + 1):
            comb[n][0] = 1
            for k_ in range(1, n + 1):
                comb[n][k_] = (comb[n-1][k_-1] + comb[n-1][k_]) % MOD

        # Precompute steps(s) for s from 1 to 800
        memo_steps = {}
        def compute_steps(s):
            if s == 1:
                return 0
            if s in memo_steps:
                return memo_steps[s]
            f_s = bin(s).count('1')
            res = 1 + compute_steps(f_s)
            memo_steps[s] = res
            return res
        for s in range(1, 801):
            compute_steps(s)

        # Compute count_power_of_two
        count_power = 0
        # Check if 1 is included (1 < n)
        if len(s) > 1 or (len(s) == 1 and s[0] == '1'):
            count_power += 1
        # Compute count_rest (number of power of two >1 and <n)
        if s != "1":
            def subtract_one(s):
                s_list = list(s)
                i = len(s_list) - 1
                while i >= 0 and s_list[i] == '0':
                    s_list[i] = '1'
                    i -= 1
                if i >= 0:
                    s_list[i] = '0'
                # Remove leading zeros
                if len(s_list) > 0 and s_list[0] == '0':
                    if len(s_list) == 1:
                        return "0"
                    else:
                        s_list = s_list[1:]
                return ''.join(s_list)
            m = subtract_one(s)
            count_rest = len(m) - 1
            count_power += count_rest
        count_power %= MOD

        # Compute count_non_power_of_two
        count_non_power = 0
        L = len(s)
        valid_s = [s_val for s_val in range(1, 801) if memo_steps.get(s_val, 0) <= k - 1]

        for s_val in valid_s:
            if s_val == 0:
                continue
            # Compute sum_comb for l=1 to L-1
            sum_comb = 0
            for l in range(1, L):
                if s_val - 1 <= l - 1:
                    sum_comb = (sum_comb + comb[l-1][s_val-1]) % MOD
            # Compute count_L using digit DP
            def compute_count_L(s_count, s_str):
                L = len(s_str)
                from functools import lru_cache
                @lru_cache(maxsize=None)
                def dp(pos, cnt, tight):
                    if pos == L:
                        return 1 if cnt == s_count else 0
                    res = 0
                    limit = int(s_str[pos]) if tight else 1
                    for b in [0, 1]:
                        if b > limit:
                            continue
                        new_tight = tight and (b == limit)
                        new_cnt = cnt + (b == 1)
                        if new_cnt > s_count:
                            continue
                        res += dp(pos + 1, new_cnt, new_tight)
                        res %= MOD
                    return res % MOD
                count = dp(0, 0, True)
                original_set_bits = s_str.count('1')
                if original_set_bits == s_count:
                    count -= 1
                return max(count, 0) % MOD
            count_L = compute_count_L(s_val, s) if L >= s_val else 0
            total_s = (sum_comb + count_L) % MOD
            count_non_power = (count_non_power + total_s) % MOD

        return (count_power + count_non_power) % MOD