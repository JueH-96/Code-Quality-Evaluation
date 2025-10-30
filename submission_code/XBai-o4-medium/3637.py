MOD = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num
        
        # Precompute factorial and inverse factorial up to 80
        max_fact = 80
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        total_sum = sum(int(c) for c in velunexorai)
        if total_sum % 2 != 0:
            return 0
        S = total_sum // 2
        n = len(velunexorai)
        E = (n + 1) // 2
        O = n // 2
        
        # Frequency of each digit
        freq = [0] * 10
        for c in velunexorai:
            d = int(c)
            freq[d] += 1
        
        # Initialize DP
        dp = [[0] * (S + 1) for _ in range(E + 1)]
        dp[0][0] = 1
        
        for d in range(10):
            current_freq = freq[d]
            if current_freq == 0:
                continue
            # Create a new temporary DP table
            temp_dp = [[0] * (S + 1) for _ in range(E + 1)]
            for k in range(E + 1):
                for s in range(S + 1):
                    if dp[k][s] == 0:
                        continue
                    max_c = min(current_freq, E - k)
                    for c in range(0, max_c + 1):
                        new_k = k + c
                        new_s = s + d * c
                        if new_s > S:
                            continue
                        term = (inv_fact[c] * inv_fact[current_freq - c]) % MOD
                        temp_dp[new_k][new_s] = (temp_dp[new_k][new_s] + dp[k][s] * term) % MOD
            # Update dp to be the new temp_dp
            dp = temp_dp
        
        return (dp[E][S] * fact[E] % MOD) * fact[O] % MOD