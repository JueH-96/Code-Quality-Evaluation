import sys
sys.setrecursionlimit(1 << 25)


# ---------- Fenwick Tree ----------
class BIT:
    __slots__ = ('n', 'bit')
    def __init__(self, n):
        self.n = n
        self.bit = [0]*(n+1)
    def add(self, idx, val=1):
        while idx <= self.n:
            self.bit[idx] += val
            idx += idx & -idx
    def sum(self, idx):
        s = 0
        while idx:
            s += self.bit[idx]
            idx -= idx & -idx
        return s
    def range_sum(self, l, r):
        return self.sum(r) - self.sum(l-1)


# ---------- main ----------
def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)

    N = int(next(it))
    A = [0] + [int(next(it)) for _ in range(N)]       # 1-indexed

    # pre-compute g[j] = rightmost i < j with 2*A[i] ≤ A[j]
    g = [0]*(N+1)
    p = 1
    for j in range(1, N+1):
        while p < j and A[p]*2 <= A[j]:
            p += 1
        g[j] = p-1        # might be 0 (means “does not exist”)

    # sort (g , j) descending by g
    order = sorted(range(1, N+1), key=lambda j: g[j], reverse=True)
    pos_in_order = 0                      # pointer inside 'order'

    Q = int(next(it))

    # store queries grouped by L
    queries_by_L = [[] for _ in range(N+2)]     # (R, index, length)
    answers = [0]*Q
    for qi in range(Q):
        L = int(next(it))
        R = int(next(it))
        queries_by_L[L].append((R, qi, R-L+1))

    bit = BIT(N)

    # process L from N down to 1
    for L in range(N, 0, -1):
        # insert every j with g[j] >= L
        while pos_in_order < N and g[order[pos_in_order]] >= L:
            bit.add(order[pos_in_order], 1)
            pos_in_order += 1

        # answer all queries that start with this L
        for R, qidx, length in queries_by_L[L]:
            T = bit.range_sum(L, R)
            answers[qidx] = min(length // 2, T)

    sys.stdout.write('
'.join(map(str, answers)))


if __name__ == "__main__":
    main()