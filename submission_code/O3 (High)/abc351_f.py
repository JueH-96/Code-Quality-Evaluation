import sys


class Fenwick:
    """1-based Fenwick Tree (BIT) supporting prefix sums"""
    __slots__ = ('n', 'bit')

    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, idx: int, value: int) -> None:
        """add `value` at position `idx`"""
        while idx <= self.n:
            self.bit[idx] += value
            idx += idx & -idx

    def sum(self, idx: int) -> int:
        """prefix sum of [1 .. idx] (0 if idx == 0)"""
        res = 0
        while idx:
            res += self.bit[idx]
            idx -= idx & -idx
        return res


def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    a = list(map(int, data[1:1 + n]))

    # coordinate compression
    uniq = sorted(set(a))
    comp = {v: i + 1 for i, v in enumerate(uniq)}      # 1-based index

    m = len(uniq)
    bit_cnt = Fenwick(m)   # stores counts of past elements
    bit_sum = Fenwick(m)   # stores sum  of past elements

    ans = 0
    for v in a:
        idx = comp[v]

        # all previous A_i that are strictly smaller than v
        cnt_less = bit_cnt.sum(idx - 1)
        sum_less = bit_sum.sum(idx - 1)

        # contribute (v - A_i) for each such A_i
        ans += cnt_less * v - sum_less

        # insert current element
        bit_cnt.add(idx, 1)
        bit_sum.add(idx, v)

    print(ans)


if __name__ == "__main__":
    main()