from math import sqrt
import sys

class Combination():
    # 10**6程度までOK
    def __init__(self, N_max, mod=998244353):
        self.mod = mod
        self.inv = [None] * (N_max + 1)
        self.fact = [None] * (N_max + 1)
        self.factinv = [None] * (N_max + 1)

        self.inv[1], self.fact[0], self.fact[1] = 1, 1, 1

        N = N_max
        for i in range(2, N + 1):
            self.inv[i] = self.mod - self.inv[self.mod % i] * (self.mod // i) % self.mod
            self.fact[i] = self.fact[i - 1] * i % self.mod
            self.factinv[i] = self.factinv[i - 1] * self.inv[i] % self.mod

    def combination(self, n, r):
        k = 1 if n < r else self.fact[n] * (self.factinv[r] * self.factinv[n - r] % self.mod) % self.mod
        return k

    def permutation(self, n, r):  # permutations
        k = 0 if n < r else self.fact[n] * self.factinv[n - r] % self.mod
        return k

    def combination_with_replacement(self, n, r):  # 繰り返し組み合わせ
        k = self.combination(r + n - 1, n - 1)
        return k

# --- main code ---
input = sys.stdin.readline
fin = [0]
fin.extend([1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 0])
fact3 = [fin[i] * 3**i % 998244353 for i in range(14)]
num0to = [i // 2 for i in fin]
num1to = [i - i // 2 for i in fin]
divisor_dict = {}
COMB = Combination(10**5)

def cal(n):
    if n > divisor_dict["MAX"]:
        exponent = int(sqrt(n))
        for i in range(exponent):
            numb = 3**i
            k = n // numb
            divisor = (num0to[k] + num1to[k]) * numb
            res[i] += divisor
    else:
        return 3**divisor_dict[n]

def MAIN():
    N, M = map(int, input().split())
    res = [0] * 14
    if N >= 3**M:
        res = [1] + [0] * 13
        ans = COMB.combination_with_replacement(M + M - 1, M - 1)
    else:
        divisor_dict["MAX"] = max([N, 3**M])
        for num in range(1, divisor_dict["MAX"] + 1):
            i, t = 0, 1
            expp = 0
            while t * 3 <= num:
                t *= 3
                i += 1
                expp = num // t
            divisor = 0
            for j in range(expp + 1):
                k = num // t**j
                divisor += COMB.combination(j + k - 1, j) * num1to[k]
            divisor_dict[num] = divisor
        cal(N)
        for num in range(1, M):
            i = num // 2
            t = 1 if i == 0 else 3**i
            cal(N // t)
        ans = sum(res) % 998244353

    print(ans)

MAIN()