import sys

MOD = 998244353


def build_min_r(n, constraints):
    """
    min_r[x][l] … for the position x and a left border l (1-indexed)
                 the minimum R among all constraints
                 (L, R, X=x) with L ≥ l.
                 If no such constraint exists the value is n+1 (acts as +∞).
    """
    inf = n + 1
    min_r = [[inf] * (n + 2) for _ in range(n + 1)]  # 1 … n  (index 0 is unused)

    for L, R, X in constraints:
        if R < min_r[X][L]:
            min_r[X][L] = R

    # propagate minimums to the left
    for x in range(1, n + 1):
        for l in range(n - 1, 0, -1):
            if min_r[x][l] > min_r[x][l + 1]:
                min_r[x][l] = min_r[x][l + 1]

    return min_r


def precompute_binom(n, mod):
    """
    binom[a][b] = C(a, b)  for 0 ≤ b ≤ a ≤ n
    """
    binom = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        binom[i][0] = binom[i][i] = 1
        for j in range(1, i):
            binom[i][j] = (binom[i - 1][j - 1] + binom[i - 1][j]) % mod
    return binom


def solve(n, constraints):
    min_r = build_min_r(n, constraints)
    binom = precompute_binom(n, MOD)

    # dp[l][r]  (1-indexed, l ≤ r) : number of valid permutations in the segment [l, r]
    # an empty segment is 1 by convention
    dp = [[0] * (n + 2) for _ in range(n + 2)]
    for i in range(1, n + 2):
        dp[i][i - 1] = 1  # empty

    for length in range(1, n + 1):
        for l in range(1, n - length + 2):
            r = l + length - 1
            total = 0
            for p in range(l, r + 1):
                # pivot p is forbidden iff some constraint (L,R,X=p) is fully contained in [l,r]
                if min_r[p][l] <= r:
                    continue
                left_cnt = dp[l][p - 1]
                right_cnt = dp[p + 1][r]
                ways = left_cnt * right_cnt % MOD
                ways = ways * binom[length - 1][p - l] % MOD  # distribute values
                total += ways
                if total >= MOD:
                    total -= MOD
            dp[l][r] = total
    return dp[1][n]


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    n, m = data[:2]
    constraints = [tuple(data[i:i + 3]) for i in range(2, 3 * m + 2, 3)]
    print(solve(n, constraints) % MOD)


if __name__ == "__main__":
    main()