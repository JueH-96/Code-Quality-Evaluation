import sys
from math import gcd

# ---------- floor_sum ----------
# (taken from AtCoder Library, public domain)
# returns sum_{i=0}^{n-1} floor((a*i+b)/m)
def floor_sum(n: int, m: int, a: int, b: int) -> int:
    """Sum of floors: Σ floor((a*i + b) / m) for i = 0 .. n-1"""
    res = 0
    while True:
        if a >= m:
            res += (n - 1) * n * (a // m) // 2
            a %= m
        if b >= m:
            res += n * (b // m)
            b %= m
        y_max = a * n + b
        if y_max < m:
            break
        n, b = divmod(y_max, m)
        a, m = m, a
    return res


# ---------- main solution ----------
def main() -> None:
    read = sys.stdin.readline

    N, M, C, K = map(int, read().split())
    A = list(map(int, read().split()))

    # construct B = (-A_i) mod M = (M - A_i) % M
    B = [(M - a) % M for a in A]
    B.sort()                       # ascending
    b_last = B[-1]                 # largest B_j
    b_first = B[0]                 # smallest B_j

    # pre-compute F0 = Σ floor((C*k)/M)
    F0 = floor_sum(K, M, C, 0)     # floor_sum(K, M, C, 0)

    # helper: count of k in [0,K) with (C*k mod M) < T
    # formula: cnt = K - (floor_sum(K, M, C, M-T) - F0)
    def count_less_than(T: int) -> int:
        if T <= 0:
            return 0
        if T >= M:
            return K
        D = floor_sum(K, M, C, M - T) - F0
        return K - D

    # prefix counts for each B_j
    prefix_cnt = [count_less_than(b) for b in B]

    cnt_interval_0   = prefix_cnt[0]                       # [0, b_first)
    cnt_interval_last = K - prefix_cnt[-1]                 # [b_last, M)

    # sum over intervals with their corresponding previous B
    sum_prev = b_last * (cnt_interval_0 + cnt_interval_last)
    for idx in range(N - 1):                               # intervals [B_i, B_{i+1})
        cnt_here = prefix_cnt[idx + 1] - prefix_cnt[idx]
        sum_prev += B[idx] * cnt_here

    # total sum of s_k = Σ (C*k mod M) for k = 0 .. K-1
    total_arith = C * K * (K - 1) // 2                     # Σ C*k
    total_s     = total_arith - M * F0                     # subtract multiples of M

    # final answer:
    #   Σ diff = Σ s_k - Σ prevB + M * cnt_interval_0
    ans = total_s - sum_prev + M * cnt_interval_0
    print(ans)


if __name__ == "__main__":
    main()