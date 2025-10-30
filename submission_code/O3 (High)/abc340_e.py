import sys


class Fenwick:
    # 0-indexed point query  /  range add
    def __init__(self, n: int):
        self.n = n + 2                   # one spare cell
        self.bit = [0] * (self.n + 1)

    def _add(self, idx: int, delta: int):
        i = idx
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def range_add(self, l: int, r: int, delta: int):
        # add delta to interval [l, r]   (0 ≤ l ≤ r < n)
        self._add(l + 1, delta)
        self._add(r + 2, -delta)

    def point_query(self, idx: int) -> int:
        s = 0
        i = idx + 1
        while i:
            s += self.bit[i]
            i -= i & -i
        return s


def main() -> None:
    it = map(int, sys.stdin.buffer.read().split())
    N = next(it)
    M = next(it)

    base = [next(it) for _ in range(N)]      # own part of every box
    B_list = [next(it) for _ in range(M)]    # operations

    bit = Fenwick(N)
    G = 0                                    # uniform part

    for B in B_list:
        extra_b = bit.point_query(B)
        K = base[B] + G + extra_b            # balls taken out

        q, r = divmod(K, N)
        G += q

        # set new value of box B  (must become q)
        base[B] = q - G - extra_b

        if r:
            l = (B + 1) % N
            e = (B + r) % N
            if l <= e:                       # no wrap
                bit.range_add(l, e, 1)
            else:                            # wraps around
                bit.range_add(l, N - 1, 1)
                bit.range_add(0, e, 1)

    # produce answer
    out = []
    for i in range(N):
        val = base[i] + G + bit.point_query(i)
        out.append(str(val))

    sys.stdout.write(' '.join(out))


if __name__ == "__main__":
    main()