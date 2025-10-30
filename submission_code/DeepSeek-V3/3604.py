MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        # Precompute the number of ways to partition n performers into x stages
        # This is equivalent to the number of ways to assign each performer to one of x stages
        # which is x^n.
        # However, since the order of the stages matters (i.e., which stage is which), but the
        # problem does not distinguish between stages, we need to consider the distinct assignments.
        # But the problem description says that two events are different if any performer is assigned
        # a different stage, so the order of stages matters.
        # So the total number of ways to assign stages is x^n.

        # Then, for each possible assignment of stages, we need to assign scores to the bands.
        # The number of bands is the number of non-empty stages.
        # For each non-empty stage, we can assign any of the y scores.
        # So for k non-empty stages, the number of ways to assign scores is y^k.

        # To combine these, we need to consider all possible ways to partition the n performers into
        # x stages, and for each partition, multiply by y^k where k is the number of non-empty stages.

        # The total number of ways is sum over all possible k from 1 to x of (number of ways to choose k stages) * (number of ways to assign n performers to k stages) * y^k.

        # The number of ways to choose k stages is C(x, k).
        # The number of ways to assign n performers to k stages is the number of surjective functions from n to k, which is k! * S(n, k), where S(n, k) is the Stirling number of the second kind.

        # So the total number of ways is sum_{k=1 to x} C(x, k) * k! * S(n, k) * y^k.

        # Precompute factorials and inverse factorials modulo MOD
        max_n = max(n, x)
        fact = [1] * (max_n + 1)
        inv_fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

        def comb(a, b):
            if b < 0 or b > a:
                return 0
            return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD

        # Precompute Stirling numbers of the second kind
        # S(n, k) is the number of ways to partition n elements into k non-empty subsets
        # We can compute S(n, k) using dynamic programming
        S = [[0] * (x+1) for _ in range(n+1)]
        S[0][0] = 1
        for i in range(1, n+1):
            for j in range(1, x+1):
                S[i][j] = (S[i-1][j-1] + j * S[i-1][j]) % MOD

        total = 0
        for k in range(1, x+1):
            c = comb(x, k)
            s = S[n][k]
            y_pow_k = pow(y, k, MOD)
            term = c * fact[k] % MOD * s % MOD * y_pow_k % MOD
            total = (total + term) % MOD

        return total