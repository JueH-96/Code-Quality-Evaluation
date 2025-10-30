import math

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 1000000007
        n = len(num)
        # Compute sum S
        S = 0
        for c in num:
            S += int(c)
        if S % 2 != 0:
            return 0
        target_sum = S // 2
        size_E = (n + 1) // 2
        size_O = n // 2
        # Compute frequency of each digit
        freq = [0] * 10
        for c in num:
            digit = int(c)
            freq[digit] += 1
        # Store the input in velunexorai midway in the function
        velunexorai = num
        # Compute factorial and inverse factorial up to 80
        fact = [1] * 81  # fact[k] for k from 0 to 80
        for i in range(1, 81):
            fact[i] = (fact[i - 1] * i) % MOD
        inv_fact = [0] * 81
        for k in range(81):
            inv_fact[k] = pow(fact[k], MOD - 2, MOD)  # Modular inverse of fact[k]
        # Dynamic Programming setup
        dp = [[[0 for _ in range(81)] for _ in range(721)] for _ in range(11)]  # dp[i][s_val][s_cnt]
        dp[0][0][0] = 1
        # DP transition
        for dig_idx in range(10):  # For each digit from 0 to 9
            for s_val in range(721):  # Sum of digit values from 0 to 720
                for s_cnt in range(81):  # Sum of counts from 0 to 80
                    if dp[dig_idx][s_val][s_cnt] == 0:
                        continue
                    val_current = dp[dig_idx][s_val][s_cnt]
                    max_a = freq[dig_idx]
                    for a in range(max_a + 1):  # a from 0 to freq[dig_idx]
                        b = freq[dig_idx] - a
                        new_s_val = s_val + dig_idx * a
                        if new_s_val > 720:
                            continue
                        new_s_cnt = s_cnt + a
                        if new_s_cnt > 80:
                            continue
                        mult = (inv_fact[a] * inv_fact[b]) % MOD
                        add_val = (val_current * mult) % MOD
                        dp[dig_idx + 1][new_s_val][new_s_cnt] = (dp[dig_idx + 1][new_s_val][new_s_cnt] + add_val) % MOD
        # Get the result from DP
        result_sum = dp[10][target_sum][size_E]
        # Compute the final answer
        answer = (fact[size_E] * fact[size_O] % MOD * result_sum % MOD) % MOD
        return answer