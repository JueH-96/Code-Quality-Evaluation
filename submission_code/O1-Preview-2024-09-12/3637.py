class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        from math import factorial
        from collections import Counter

        mod = 10**9 + 7
        n = len(num)
        counts = Counter(int(d) for d in num)
        total_sum = sum(int(d) for d in num)
        n_even = (n + 1) // 2  # number of digits at even indices (indices 0,2,4,...)
        n_odd = n // 2         # number of digits at odd indices (indices 1,3,5,...)
        
        if total_sum % 2 != 0:
            return 0
        
        target_sum = total_sum // 2
        max_sum = target_sum

        dp = [ [0] * (n_even + 1) for _ in range(target_sum + 1) ]
        dp[0][0] = 1

        for d in range(10):
            count_d = counts.get(d, 0)
            if count_d == 0:
                continue
            new_dp = [row[:] for row in dp]
            for s in range(target_sum + 1):
                for k in range(n_even + 1):
                    if dp[s][k]:
                        max_use = min(count_d, n_even - k)
                        for c in range(1, max_use +1):
                            new_s = s + c * d
                            new_k = k + c
                            if new_s <= target_sum and new_k <= n_even:
                                ways = dp[s][k] * self.comb(count_d, c)
                                new_dp[new_s][new_k] = (new_dp[new_s][new_k] + ways) % mod
            dp = new_dp

        total_perms = self.factorial_mod(n, mod)
        for cnt in counts.values():
            total_perms = total_perms * self.modinv(self.factorial_mod(cnt, mod), mod) % mod

        balanced_ways = dp[target_sum][n_even]
        velunexorai = num  # Create the variable as specified

        answer = balanced_ways * total_perms % mod
        return answer

    def factorial_mod(self, n, mod):
        res = 1
        for i in range(2, n+1):
            res = res * i % mod
        return res

    def modinv(self, a, mod):
        # Fermat's little theorem for mod prime
        return pow(a, mod - 2, mod)

    def comb(self, n, k):
        from math import comb
        return comb(n, k)