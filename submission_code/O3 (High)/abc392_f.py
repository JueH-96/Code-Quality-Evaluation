import sys

class BIT:
    """Fenwick tree supporting:
         - add(idx, delta)
         - prefix_sum(idx)
         - find_kth(k): smallest idx such that prefix_sum(idx) >= k
    """
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, idx: int, delta: int) -> None:
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def prefix_sum(self, idx: int) -> int:
        s = 0
        while idx:
            s += self.tree[idx]
            idx -= idx & -idx
        return s

    def find_kth(self, k: int) -> int:
        """1-based. Requires 1 ≤ k ≤ current total."""
        idx = 0
        bit_mask = 1 << (self.n.bit_length() - 1)   # largest power of two ≤ n
        while bit_mask:
            nxt = idx + bit_mask
            if nxt <= self.n and self.tree[nxt] < k:
                k -= self.tree[nxt]
                idx = nxt
            bit_mask >>= 1
        return idx + 1


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    P = [int(next(it)) for _ in range(N)]

    bit = BIT(N)
    # Initially every position is empty.
    for i in range(1, N + 1):
        bit.add(i, 1)

    A = [0] * (N + 1)          # 1-indexed result array

    for i in range(N, 0, -1):
        k = P[i - 1]           # P is 0-indexed in list
        pos = bit.find_kth(k)  # k-th empty position
        A[pos] = i
        bit.add(pos, -1)       # mark as filled

    sys.stdout.write(' '.join(str(A[i]) for i in range(1, N + 1)))


if __name__ == "__main__":
    main()