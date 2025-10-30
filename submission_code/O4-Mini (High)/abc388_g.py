import sys
import threading

def main():
    import sys
    data = sys.stdin
    # Read N and array A
    line = data.readline().split()
    while not line:
        line = data.readline().split()
    N = int(line[0])
    A_list = list(map(int, data.readline().split()))
    # 1-index A
    A = [0] * (N + 1)
    for i in range(N):
        A[i+1] = A_list[i]

    # Precompute LB[i]: the smallest j > i with A[j] >= 2*A[i], or N+1 if none
    LB = [0] * (N + 2)
    j = 1
    for i in range(1, N+1):
        if j <= i:
            j = i + 1
        # advance j until A[j] >= 2*A[i]
        # j may go up to N+1
        while j <= N and A[j] < 2 * A[i]:
            j += 1
        LB[i] = j  # if j == N+1, it denotes no valid match

    # Precompute D[i] = LB[i] - i
    D = [0] * (N + 1)
    for i in range(1, N+1):
        D[i] = LB[i] - i

    # Build log table
    import math
    LOG = [0] * (N + 1)
    for i in range(2, N+1):
        LOG[i] = LOG[i >> 1] + 1
    K = LOG[N] + 1

    # Build sparse table for range maximum of D
    # st[k][i] = max of D[i..i+2^k-1]
    st = [[0] * (N + 1) for _ in range(K)]
    # k = 0
    for i in range(1, N+1):
        st[0][i] = D[i]
    # higher levels
    j = 1
    length = 2
    while length <= N:
        half = length >> 1
        row = st[j]
        prev = st[j-1]
        for i in range(1, N - length + 2):
            # max over [i..i+length-1] = max(prev[i..i+half-1], prev[i+half..i+length-1])
            v = prev[i]
            w = prev[i + half]
            if w > v:
                v = w
            row[i] = v
        j += 1
        length <<= 1

    def range_max(l, r):
        # inclusive l,r, 1-indexed
        length = r - l + 1
        k = LOG[length]
        row = st[k]
        # max over [l..r] = max(row[l], row[r-2^k+1])
        end = r - (1 << k) + 1
        v = row[l]
        w = row[end]
        if w > v:
            v = w
        return v

    # Process queries
    out = []
    Q = int(data.readline().strip())
    for _ in range(Q):
        parts = data.readline().split()
        while not parts:
            parts = data.readline().split()
        L = int(parts[0])
        R = int(parts[1])
        B = R - L + 1
        # max possible pairs
        high = B >> 1  # floor(B/2)
        low = 0
        ans = 0
        # binary search max t in [0..high] s.t. max(D[L..L+t-1]) <= B - t
        # note for t=0: empty range, always true
        while low <= high:
            mid = (low + high) >> 1
            if mid == 0:
                # always feasible
                ans = 0
                low = 1
                continue
            # compute max D over [L..L+mid-1]
            end_i = L + mid - 1
            # ensure end_i <= N
            # by construction mid <= floor(B/2) ensures end_i <= R-1 < N
            maxD = range_max(L, end_i)
            # feasibility: maxD <= B - mid
            if maxD <= B - mid:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()