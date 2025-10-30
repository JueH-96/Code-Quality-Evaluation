MOD = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        num_str = num
        max_n = 80
        
        # Precompute factorials and inverse factorials
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Count the occurrences of each digit
        from collections import defaultdict
        counts = defaultdict(int)
        for c in num_str:
            counts[int(c)] += 1
        
        # Convert counts to a list where index represents the digit (0-9)
        counts_list = [0] * 10
        for d in counts:
            counts_list[d] = counts[d]
        
        N = len(num_str)
        E = (N + 1) // 2
        O = N // 2
        total_sum = sum(int(c) for c in num_str)
        
        if total_sum % 2 != 0:
            return 0
        
        target = total_sum // 2
        
        # Initialize DP table
        dp = [[0] * (target + 1) for _ in range(E + 1)]
        dp[0][0] = 1
        
        for d in range(10):
            c = counts_list[d]
            new_dp = [[0] * (target + 1) for _ in range(E + 1)]
            for t in range(E + 1):
                for s in range(target + 1):
                    if dp[t][s] == 0:
                        continue
                    max_t_d = min(c, E - t)
                    for t_d in range(0, max_t_d + 1):
                        new_t = t + t_d
                        new_s = s + d * t_d
                        if new_t > E or new_s > target:
                            continue
                        # Contribution calculation
                        contribution = fact[c] if c != 0 else 1
                        inv_t_sq = pow(inv_fact[t_d], 2, MOD) if t_d != 0 else 1
                        rem = c - t_d
                        inv_rem_sq = pow(inv_fact[rem], 2, MOD) if rem != 0 else 1
                        contribution = contribution * inv_t_sq % MOD
                        contribution = contribution * inv_rem_sq % MOD
                        new_dp[new_t][new_s] = (new_dp[new_t][new_s] + dp[t][s] * contribution) % MOD
            dp = new_dp
        
        E_fact = fact[E] if E <= max_n else 0
        O_fact = fact[O] if O <= max_n else 0
        total = (E_fact * O_fact) % MOD
        total = total * dp[E][target] % MOD
        return total