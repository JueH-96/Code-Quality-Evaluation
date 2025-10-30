import sys
sys.setrecursionlimit(1 << 25)


class FenwickMax:
    """Fenwick tree with max operation (1-index based)"""
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def update(self, idx: int, value: int):
        """set bit[idx] = max(bit[idx] , value)"""
        n = self.n
        while idx <= n:
            if value > self.bit[idx]:
                self.bit[idx] = value
            idx += idx & -idx

    def pref_max(self, idx: int) -> int:
        """max on prefix [1 .. idx]"""
        res = 0
        while idx:
            cur = self.bit[idx]
            if cur > res:
                res = cur
            idx -= idx & -idx
        return res


def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    Q = int(next(it))

    A = [0] * N
    for i in range(N):
        A[i] = int(next(it))

    # pair each array element with its index (1-based)
    pairs = [(A[i], i + 1) for i in range(N)]
    pairs.sort()                                   # by value

    # read queries
    queries = []
    for qid in range(Q):
        R = int(next(it))
        X = int(next(it))
        queries.append((X, R, qid))
    queries.sort()                                 # by X

    fenwick = FenwickMax(N)
    answers = [0] * Q

    p = 0                    # next array element to insert
    total = len(pairs)

    for X, R, qid in queries:
        # insert all array elements whose value ≤ X
        while p < total and pairs[p][0] <= X:
            v = pairs[p][0]               # current value
            # collect full block with the same value
            same_val = []
            while p < total and pairs[p][0] == v:
                idx = pairs[p][1]
                best = fenwick.pref_max(idx - 1)
                same_val.append((idx, best + 1))
                p += 1
            # write them afterwards
            for idx, val in same_val:
                fenwick.update(idx, val)

        # answer the query
        answers[qid] = fenwick.pref_max(R)

    # restore original order
    sys.stdout.write('
'.join(map(str, answers)))


if __name__ == "__main__":
    main()