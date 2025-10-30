class Solution:
    MOD = 10**9 + 7
    max_n = 1000

    # Precompute Stirling numbers of the second kind
    _stirling = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    _stirling[0][0] = 1
    for n in range(1, max_n + 1):
        for k in range(1, n + 1):
            _stirling[n][k] = (k * _stirling[n-1][k] + _stirling[n-1][k-1]) % MOD

    # Precompute factorials and inverse factorials mod MOD
    _fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        _fact[i] = _fact[i-1] * i % MOD

    _inv_fact = [1] * (max_n + 1)
    _inv_fact[max_n] = pow(_fact[max_n], MOD - 2, MOD)
    for i in range(max_n - 1, -1, -1):
        _inv_fact[i] = _inv_fact[i + 1] * (i + 1) % MOD

    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = Solution.MOD
        res = 0
        stirling = Solution._stirling
        fact = Solution._fact
        inv_fact = Solution._inv_fact

        for k in range(1, x + 1):
            if k > n:
                s = 0
            else:
                s = stirling[n][k]
            if s == 0:
                continue
            # Calculate combination C(x, k)
            comb = (fact[x] * inv_fact[k] % MOD) * inv_fact[x - k] % MOD
            # Calculate the term for current k
            term = comb * fact[k] % MOD
            term = term * s % MOD
            term = term * pow(y, k, MOD) % MOD
            res = (res + term) % MOD

        return res