class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        max_fact = 80
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        def comb(n, k):
            if k < 0 or k > n:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
        
        freq = [0] * 10
        for c in num:
            freq[int(c)] += 1
        
        total_sum = sum(int(c) for c in num)
        n = len(num)
        m = (n + 1) // 2
        k = n // 2
        
        if total_sum % 2 != 0:
            return 0
        
        target_sum = total_sum // 2
        
        # Check if target_sum is feasible
        if target_sum < 0 or target_sum > 9 * m:
            return 0
        
        # Calculate D = product of factorials of frequencies
        D = 1
        for d in range(10):
            D = D * fact[freq[d]] % MOD
        
        # Initialize DP table
        dp = [[0] * (target_sum + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        
        for d in range(10):
            current_freq = freq[d]
            if current_freq == 0:
                continue
            
            next_dp = [[0] * (target_sum + 1) for _ in range(m + 1)]
            for count in range(m + 1):
                for current_sum in range(target_sum + 1):
                    if dp[count][current_sum] == 0:
                        continue
                    max_take = min(current_freq, m - count)
                    for take in range(max_take + 1):
                        new_count = count + take
                        new_sum = current_sum + take * d
                        if new_sum > target_sum:
                            continue
                        c = comb(current_freq, take)
                        next_dp[new_count][new_sum] = (next_dp[new_count][new_sum] + dp[count][current_sum] * c) % MOD
            dp = next_dp
        
        sum_dp = dp[m][target_sum]
        if sum_dp == 0:
            return 0
        
        result = fact[m] * fact[k] % MOD
        result = result * sum_dp % MOD
        result = result * pow(D, MOD-2, MOD) % MOD
        
        return result