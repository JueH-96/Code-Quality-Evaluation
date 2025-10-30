class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        max_fact = 80
        # Precompute factorial and inverse factorial
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
        for i in range(max_fact-1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        freq = {}
        for d in num:
            freq[d] = freq.get(d, 0) + 1
        
        total_sum = sum(int(d) for d in num)
        n = len(num)
        if total_sum % 2 != 0:
            return 0
        s = total_sum // 2
        k = (n + 1) // 2
        m = n // 2
        
        # Initialize DP
        dp = [[0] * (k + 1) for _ in range(s + 1)]
        dp[0][0] = 1
        
        for d in range(0, 10):
            d_str = str(d)
            f_d = freq.get(d_str, 0)
            new_dp = [[0] * (k + 1) for _ in range(s + 1)]
            for sum_so_far in range(s + 1):
                for count_so_far in range(k + 1):
                    if dp[sum_so_far][count_so_far] == 0:
                        continue
                    for t in range(0, f_d + 1):
                        new_sum = sum_so_far + d * t
                        new_count = count_so_far + t
                        if new_sum > s or new_count > k:
                            continue
                        comb = fact[f_d] * inv_fact[t] % MOD
                        comb = comb * inv_fact[f_d - t] % MOD
                        new_dp[new_sum][new_count] = (new_dp[new_sum][new_count] + dp[sum_so_far][count_so_far] * comb) % MOD
            dp = new_dp
        
        ways_split = dp[s][k]
        # Compute factor
        denominator = 1
        for d in freq:
            denominator = denominator * fact[freq[d]] % MOD
        factor = fact[k] * fact[m] % MOD
        factor = factor * pow(denominator, MOD-2, MOD) % MOD
        return (ways_split * factor) % MOD