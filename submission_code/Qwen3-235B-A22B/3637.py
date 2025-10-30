class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        max_fact = 80
        # Precompute factorial and inverse factorial mod MOD
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact -1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        # Count digits and compute sum_total
        count = [0] * 10
        sum_total = 0
        for c in num:
            d = int(c)
            count[d] += 1
            sum_total += d
        
        if sum_total % 2 != 0:
            return 0
        
        s_total = sum_total // 2
        L = len(num)
        even_len = (L + 1) // 2
        odd_len = L // 2
        
        # Compute denominator: product of count[d]! mod MOD
        denominator = 1
        for d in range(10):
            cnt = count[d]
            if cnt > 0:
                denominator = denominator * fact[cnt] % MOD
        
        # Compute C = (even_len! * odd_len! ) / denominator mod MOD
        inv_denominator = pow(denominator, MOD-2, MOD)
        C = fact[even_len] * fact[odd_len] % MOD
        C = C * inv_denominator % MOD
        
        # Initialize DP
        dp = [[0] * (s_total + 1) for _ in range(even_len + 1)]
        dp[0][0] = 1
        
        for d_val in range(10):
            cnt = count[d_val]
            if cnt == 0:
                continue
            
            # Create new_dp initialized to 0
            new_dp = [[0] * (s_total + 1) for _ in range(even_len + 1)]
            for k in range(even_len + 1):
                for s in range(s_total + 1):
                    if dp[k][s] == 0:
                        continue
                    
                    # Try all possible x for current digit d
                    max_x = min(cnt, even_len - k)
                    for x in range(0, max_x + 1):
                        new_k = k + x
                        new_s = s + d_val * x
                        if new_s > s_total or new_k > even_len:
                            continue
                        
                        # Compute binomial coefficient C(cnt, x)
                        if x < 0 or x > cnt:
                            binom = 0
                        else:
                            binom = fact[cnt] * inv_fact[x] % MOD
                            binom = binom * inv_fact[cnt - x] % MOD
                        
                        new_dp[new_k][new_s] = (new_dp[new_k][new_s] + dp[k][s] * binom) % MOD
            
            dp = new_dp
        
        S = dp[even_len][s_total]
        ans = (C * S) % MOD
        return ans