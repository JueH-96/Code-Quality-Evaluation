import sys
import math

def main():
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Precompute prefix sums for '1's and '2's
    pre1 = [0] * (N + 1)
    pre2 = [0] * (N + 1)
    for i in range(1, N + 1):
        c = S[i-1]
        pre1[i] = pre1[i-1] + (1 if c == '1' else 0)
        pre2[i] = pre2[i-1] + (1 if c == '2' else 0)

    # Precompute prefix sum for '/'
    pre_slash = [0] * (N + 1)
    for i in range(1, N + 1):
        c = S[i-1]
        pre_slash[i] = pre_slash[i-1] + (1 if c == '/' else 0)

    # Build the v array
    v = [-float('inf')] * N
    for i in range(N):
        if S[i] == '/':
            v[i] = pre1[i] - (pre2[N] - pre2[i+1])

    # Precompute log_table for Sparse Table
    log_table = [0] * (N + 1)
    for i in range(2, N + 1):
        log_table[i] = log_table[i // 2] + 1

    # Build Sparse Table for v
    k_max = log_table[N] + 1 if N > 0 else 0
    st = []
    if k_max > 0:
        st = [[0] * N for _ in range(k_max)]
        st[0] = v.copy()
        for k in range(1, k_max):
            for i in range(N - (1 << k) + 1):
                st[k][i] = max(st[k-1][i], st[k-1][i + (1 << (k-1))])

    # Process each query
    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())
        L0 = L - 1
        R0 = R - 1
        sum_has = pre_slash[R] - pre_slash[L0]
        if sum_has == 0:
            print(0)
            continue

        # Calculate base
        base = (pre2[R] - pre2[L0]) + 1

        # Calculate maximum v in [L0, R0]
        length = R0 - L0 + 1
        if length == 0:
            print(0)
            continue

        k = log_table[length]
        a = st[k][L0] if k < k_max else -float('inf')
        b_pos = R0 - (1 << k) + 1
        if b_pos < L0:
            b = -float('inf')
        else:
            b = st[k][b_pos] if k < k_max else -float('inf')
        max_v = max(a, b)

        ans = max(0, base + max_v)
        print(ans)

if __name__ == '__main__':
    main()