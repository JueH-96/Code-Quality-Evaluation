mod = 10**9 + 7
max_val = 1000
comb_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
for i in range(max_val + 1):
    comb_table[i][0] = 1
    for j in range(1, i + 1):
        comb_table[i][j] = (comb_table[i - 1][j - 1] + comb_table[i - 1][j]) % mod

class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        total = 0
        min_kx = min(n, x)
        for k in range(1, min_kx + 1):
            surj = 0
            for i in range(0, k + 1):
                term = comb_table[k][i] * pow(k - i, n, mod) % mod
                if i % 2 == 1:
                    term = -term
                surj = (surj + term) % mod
            surj %= mod
            if surj < 0:
                surj += mod
            c_xk = comb_table[x][k]
            y_power = pow(y, k, mod)
            total = (total + c_xk * surj % mod * y_power) % mod
        return total