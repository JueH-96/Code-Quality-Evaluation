MOD = 10**9 + 7

def comb(n, k, factorial, inv_factorial):
    if k < 0 or k > n:
        return 0
    return (factorial[n] * inv_factorial[k] % MOD) * inv_factorial[n - k] % MOD

def precompute_factorials(max_n, MOD):
    factorial = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        factorial[i] = factorial[i-1] * i % MOD
    inv_factorial = [1] * (max_n + 1)
    inv_factorial[max_n] = pow(factorial[max_n], MOD-2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv_factorial[i] = inv_factorial[i+1] * (i+1) % MOD
    return factorial, inv_factorial

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        if not sick:
            return 1
        max_n = n
        factorial, inv_factorial = precompute_factorials(max_n, MOD)
        total = 1
        prev = -1
        for i in range(len(sick)):
            current = sick[i]
            if prev == -1:
                gap = current - 0
                if gap > 0:
                    total = total * 1 % MOD
            else:
                gap = current - prev - 1
                if gap > 0:
                    total = total * pow(2, gap - 1, MOD) % MOD
            prev = current
        gap = n - 1 - sick[-1]
        if gap > 0:
            total = total * 1 % MOD
        return total