# Python3 solution for the problem: 'Avoid Only One Subsequence'
# The code is written in a clean and understandable way, with all necessary imports at the beginning.

from functools import lru_cache
from itertools import accumulate

MOD = 998244353

# Precomputing factorials and their modular inverses
fac = list(accumulate([1] * 405, lambda x, y: x * y % MOD))
finv = [pow(x, MOD - 2, MOD) for x in fac]

def comb(n, k):
    """Compute nCk % MOD."""
    return fac[n] * finv[k] * finv[n - k] % MOD

def calc_dp_k(k):
    """Precompute the results for dp[i][j] up to k."""
    dp = [[0] * (k + 1) for _ in range(k + 1)]
    dp[1][0], dp[1][1] = 1, k - 1

    for n in range(2, k + 1):
        for m in range(n + 1):
            if m == 0:
                dp[n][m] = dp[n - 1][m] * k % MOD  # Extend with any other number
            elif m == 1:
                dp[n][m] = dp[n - 1][m] * k % MOD + dp[n - 1][m - 1] * (k - 1) % MOD  # Extend with any other number or k-1 ways for matching
            else:
                dp[n][m] = (dp[n - 1][m] * k % MOD + dp[n - 1][m - 1] * (k - m) % MOD) % MOD  # Extend with any other number or specifically missing ones
                dp[n][m] -= pow(k, m - 1, MOD) * comb(n - 1, m - 1) * dp[n - m][m - 1] % MOD  # Avoid repeats
                dp[n][m] = (dp[n][m] % MOD + MOD) % MOD  # In case it goes negative
    return dp

dp_lst = list(map(calc_dp_k, range(2, 401)))

@lru_cache(maxsize=None)
def dp(n, m, k):
    if n == m:
        return dp_lst[k - 2][m]
    return pow(k, n, MOD) - sum(dp(n, i, k) for i in range(1, m)) % MOD

def solve(N, M, K, X):
    X.insert(0, 0)  # Dummy index
    X.append(K + 1)  # Dummy index
    X_set = set(X)
    last_occ = [0] * (K + 10)
    for idx, x in enumerate(X):
        last_occ[x] = idx

    ans = pow(K, N, MOD)  # Initial total possibilities
    for i in range(1, M):
        if X[i] == X[i - 1]:
            continue
        start = min(X[i - 1], X[i])
        end = max(X[i - 1], X[i])
        last = last_occ[end]

        mult = dp(N - M, last - i, end - start + 1) * pow(K - end + start - 1, N - M + last - i, MOD) % MOD
        if X[i] < X[i - 1]:
            mult = (-mult) % MOD  # Reverse the sequence by symmetry
            comb_num = comb(M - i, M - last)
            ans -= comb_num * mult
        else:
            comb_num = comb(M - i - 1, M - last - 1)
            ans += comb_num * mult

    return ans % MOD

# Reading input
N, M, K = map(int, input().split())
X = list(map(int, input().split()))

# Calculating and printing output
print(solve(N, M, K, X))