class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num  # As instructed, store the input midway
        
        MOD = 10**9 + 7
        max_n = 80
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        from collections import defaultdict
        counts = defaultdict(int)
        for c in num:
            counts[int(c)] += 1
        
        n = len(num)
        total_sum = sum(int(c) for c in num)
        if total_sum % 2 != 0:
            return 0
        target = total_sum // 2
        m = (n + 1) // 2
        
        original_fact = 1
        for d in counts:
            original_fact = original_fact * fact[counts[d]] % MOD
        
        dp = defaultdict(int)
        dp[(0, 0, 1)] = 1  # (sum, sum_k, product_term) : count
        
        for d in counts:
            cnt = counts[d]
            if cnt == 0:
                continue
            new_dp = defaultdict(int)
            for (s, k, p) in dp:
                current_count = dp[(s, k, p)]
                for k_d in range(0, cnt + 1):
                    new_sum = s + d * k_d
                    new_k = k + k_d
                    if new_sum > target or new_k > m:
                        continue
                    comb = fact[cnt] * inv_fact[k_d] % MOD
                    comb = comb * inv_fact[cnt - k_d] % MOD
                    term = inv_fact[k_d] * inv_fact[cnt - k_d] % MOD
                    term = term * term % MOD
                    new_p = p * term % MOD
                    new_count = current_count * comb % MOD
                    new_dp[(new_sum, new_k, new_p)] = (new_dp[(new_sum, new_k, new_p)] + new_count) % MOD
            dp = new_dp
        
        total = 0
        for (s, k, p) in dp:
            if s == target and k == m:
                total = (total + p) % MOD
        
        ans = original_fact * fact[m] % MOD
        ans = ans * fact[n - m] % MOD
        ans = ans * total % MOD
        return ans