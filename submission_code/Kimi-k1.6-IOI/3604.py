class Solution:
    MOD = 10**9 + 7
    max_n = 1000
    stirling = None  # Static variable to hold precomputed Stirling numbers

    @staticmethod
    def precompute():
        if Solution.stirling is not None:
            return
        max_n = Solution.max_n
        MOD = Solution.MOD
        stirling = [[0] * (max_n + 2) for _ in range(max_n + 2)]
        stirling[0][0] = 1
        for n in range(1, max_n + 1):
            for k in range(1, n + 1):
                stirling[n][k] = (k * stirling[n-1][k] + stirling[n-1][k-1]) % MOD
        Solution.stirling = stirling

    def numberOfWays(self, n: int, x: int, y: int) -> int:
        self.precompute()
        MOD = Solution.MOD
        stirling = Solution.stirling
        min_k = min(x, n)
        res = 0
        for k in range(1, min_k + 1):
            # Compute P(x, k) = x * (x-1) * ... * (x -k + 1) mod MOD
            perm = 1
            for i in range(k):
                perm = perm * (x - i) % MOD
            s = stirling[n][k]
            y_power = pow(y, k, MOD)
            term = perm * s % MOD
            term = term * y_power % MOD
            res = (res + term) % MOD
        return res