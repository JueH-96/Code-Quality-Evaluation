class Solution:
    mod = 10**9 + 7
    max_n = 1000
    comb = None

    def __init__(self):
        if Solution.comb is None:
            max_n = Solution.max_n
            mod = Solution.mod
            comb = [[0] * (max_n + 1) for _ in range(max_n + 1)]
            for i in range(max_n + 1):
                comb[i][0] = 1
                for j in range(1, i + 1):
                    comb[i][j] = (comb[i-1][j-1] + comb[i-1][j]) % mod
            Solution.comb = comb

    def numberOfWays(self, n: int, x: int, y: int) -> int:
        mod = Solution.mod
        comb = Solution.comb
        k_max = min(n, x)
        total = 0
        for k in range(1, k_max + 1):
            c1 = comb[x][k]
            term2 = 0
            for i in range(0, k + 1):
                sign = 1 if i % 2 == 0 else -1
                c2 = comb[k][i]
                base = k - i
                power_val = pow(base, n, mod)
                term_val = c2 * power_val % mod
                if sign == 1:
                    term2 = (term2 + term_val) % mod
                else:
                    term2 = (term2 - term_val) % mod
            term2 = term2 % mod
            if term2 < 0:
                term2 += mod
            term3 = pow(y, k, mod)
            total = (total + c1 * term2 % mod * term3) % mod
        return total