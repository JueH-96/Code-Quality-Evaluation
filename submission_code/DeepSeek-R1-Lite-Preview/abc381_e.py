import sys
import bisect

def main():
    import math
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    # 1-based indexing
    prefix_ones = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_ones[i] = prefix_ones[i - 1] + (1 if S[i - 1] == '1' else 0)
    suffix_twos = [0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_twos[i] = suffix_twos[i + 1] + (1 if S[i - 1] == '2' else 0)
    # Collect positions of '/'
    slashes = [i for i in range(1, N + 1) if S[i - 1] == '/']
    M = len(slashes)
    if M == 0:
        # No slashes, all queries have answer 0
        for _ in range(Q):
            print(0)
        return
    # Compute min_k for each slash
    min_k = [min(prefix_ones[slashes[i]], suffix_twos[slashes[i]]) for i in range(M)]
    # Build sparse table for range maximum queries
    log = math.floor(math.log2(M)) + 1
    st = [[0] * M for _ in range(log)]
    st[0] = min_k
    for k in range(1, log):
        for i in range(M - (1 << k) + 1):
            st[k][i] = max(st[k - 1][i], st[k - 1][i + (1 << (k - 1))])
    # Function to get max in range [l, r] (0-indexed)
    def query(l, r):
        if l > r:
            return 0
        length = r - l + 1
        k = length.bit_length() - 1
        return max(st[k][l], st[k][r - (1 << k) + 1])
    # Process queries
    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())
        # Find first slash >= L
        left = bisect.bisect_left(slashes, L)
        # Find last slash <= R
        right = bisect.bisect_right(slashes, R) - 1
        if left > right:
            print(0)
        else:
            max_k = query(left, right)
            M_val = 2 * max_k + 1
            print(M_val)

if __name__ == '__main__':
    main()