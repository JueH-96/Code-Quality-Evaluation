class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        freq = [0] * 10
        for ch in num:
            freq[int(ch)] += 1
        n = len(num)
        S = sum(int(ch) for ch in num)
        if S % 2 != 0:
            return 0
        target = S // 2
        dp = [0] * (target + 1)
        dp[0] = 1
        for d in range(10):
            if freq[d] == 0:
                continue
            digit = d
            for k in range(1, freq[d] + 1):
                val = digit * k
                if val > target:
                    break
                for j in range(target, val - 1, -1):
                    dp[j] = (dp[j] + dp[j - val]) % MOD
        factorial = [1] * (n + 1)
        for i in range(1, n + 1):
            factorial[i] = (factorial[i - 1] * i) % MOD
        inv_fact = [1] * (n + 1)
        inv_fact[n] = pow(factorial[n], MOD - 2, MOD)
        for i in range(n - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
        even_positions = (n + 1) // 2
        odd_positions = n // 2
        even_fact = factorial[even_positions]
        odd_fact = factorial[odd_positions]
        result = 0
        from math import comb
        for selection in range(target + 1):
            if dp[selection] == 0:
                continue
            freq_even = [0] * 10
            for d in range(10):
                max_k = min(freq[d], selection // d) if d != 0 else 0
                for k in range(1, max_k + 1):
                    if selection - d * k < 0:
                        break
                    dp[selection] = (dp[selection] + dp[selection - d * k] * comb(freq[d], k)) % MOD
            coeff_even = even_fact
            coeff_odd = odd_fact
            for d in range(10):
                k = freq_even[d]
                coeff_even = (coeff_even * inv_fact[k]) % MOD
                m = freq[d] - k
                coeff_odd = (coeff_odd * inv_fact[m]) % MOD
            result = (result + dp[selection] * coeff_even % MOD * coeff_odd % MOD) % MOD
        return result