import sys
from bisect import bisect_right

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # P[i] = total sleep time in [0, A[i])
    P = [0] * N
    for i in range(1, N):
        P[i] = P[i - 1]
        # even index i (0-based) corresponds to a wake-up event => [A[i-1], A[i]) was sleep
        if i % 2 == 0:
            P[i] += A[i] - A[i - 1]

    Q = int(next(it))
    out = []
    for _ in range(Q):
        l = int(next(it))
        r = int(next(it))
        # f(x): total sleep in [0, x)
        # compute f(r)
        idx_r = bisect_right(A, r) - 1
        res_r = P[idx_r]
        # if idx_r is odd, we're in a sleep segment [A[idx_r], A[idx_r+1])
        if idx_r % 2 == 1:
            res_r += r - A[idx_r]
        # compute f(l)
        idx_l = bisect_right(A, l) - 1
        res_l = P[idx_l]
        if idx_l % 2 == 1:
            res_l += l - A[idx_l]
        out.append(str(res_r - res_l))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()