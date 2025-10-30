class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num
        MOD = 10**9 + 7

        # Count digits and compute total sum
        n = len(velunexorai)
        counts = [0] * 10
        for ch in velunexorai:
            counts[int(ch)] += 1
        total = sum(d * c for d, c in enumerate(counts))
        # If total sum is odd, can't split equally
        if total % 2 != 0:
            return 0

        target = total // 2
        n_even = (n + 1) // 2
        n_odd = n // 2

        # Precompute factorials and inverse factorials up to max(n, max count)
        max_n = max(n, max(counts))
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i - 1] * i % MOD
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
        for i in range(max_n, 0, -1):
            inv_fact[i - 1] = inv_fact[i] * i % MOD

        # dp[j][k] = sum of 1/(prod e[d]! * prod(c[d]-e[d])!) over processed digits,
        # where j is how many digits picked for even indices and k is their sum
        dp = [[0] * (target + 1) for _ in range(n_even + 1)]
        dp[0][0] = 1

        # Process each digit value
        for d in range(10):
            c = counts[d]
            if c == 0:
                continue
            dp2 = [[0] * (target + 1) for _ in range(n_even + 1)]
            for j in range(n_even + 1):
                for k in range(target + 1):
                    val = dp[j][k]
                    if not val:
                        continue
                    # Try placing e copies of digit d in even slots
                    max_e = min(c, n_even - j)
                    for e in range(max_e + 1):
                        new_j = j + e
                        new_k = k + e * d
                        if new_k > target:
                            break
                        # Multiply by inv_fact[e] * inv_fact[c-e] to build
                        # 1/(e! * (c-e)!)
                        contrib = val * inv_fact[e] % MOD * inv_fact[c - e] % MOD
                        dp2[new_j][new_k] = (dp2[new_j][new_k] + contrib) % MOD
            dp = dp2

        # Multiply by even! * odd! to count permutations within slots
        result = dp[n_even][target] * fact[n_even] % MOD * fact[n_odd] % MOD
        return result