mod = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        n = len(num)
        n_even = (n + 1) // 2
        n_odd = n // 2
        total_sum = sum(int(c) for c in num)
        
        if total_sum % 2 != 0:
            return 0
        half = total_sum // 2
        
        max_n = 100
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % mod
        
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], mod - 2, mod)
        for i in range(max_n, 0, -1):
            inv_fact[i-1] = inv_fact[i] * i % mod
        
        freq = [0] * 10
        for c in num:
            freq[int(c)] += 1
        
        dp = [[0] * (half + 1) for _ in range(n_even + 1)]
        dp[0][0] = 1
        
        for d in range(10):
            if freq[d] == 0:
                continue
            new_dp = [[0] * (half + 1) for _ in range(n_even + 1)]
            for c in range(n_even + 1):
                for s in range(half + 1):
                    if dp[c][s] == 0:
                        continue
                    max_k = min(freq[d], n_even - c)
                    for k in range(max_k + 1):
                        new_sum = s + k * d
                        if new_sum > half:
                            break
                        factor = inv_fact[k] * inv_fact[freq[d] - k] % mod
                        new_dp[c + k][new_sum] = (new_dp[c + k][new_sum] + dp[c][s] * factor) % mod
            dp = new_dp
        
        result = fact[n_even] * fact[n_odd] % mod
        result = result * dp[n_even][half] % mod
        return result