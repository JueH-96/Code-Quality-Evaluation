def solve():
    N, K = map(int, input().split())
    P = list(map(int, input().split()))
    MOD = 998244353

    def power(a, b):
        res = 1
        a %= MOD
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % MOD
            a = (a * a) % MOD
            b //= 2
        return res

    def inverse(n):
        return power(n, MOD - 2)

    expected_inversions = 0
    denominator = N - K + 1
    inv_denominator = inverse(denominator)
    inv_2 = inverse(2)

    for a in range(N):
        for b in range(a + 1, N):
            prob_both_affected_num = max(0, a - b + K)
            prob_both_affected = (prob_both_affected_num * inv_denominator) % MOD

            if P[a] > P[b]:
                expected_contribution = (1 - (prob_both_affected * inv_2) % MOD + MOD) % MOD
            else:
                expected_contribution = (prob_both_affected * inv_2) % MOD

            expected_inversions = (expected_inversions + expected_contribution) % MOD

    print(int(expected_inversions))

solve()