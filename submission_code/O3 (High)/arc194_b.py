import sys


class BIT:
    """Fenwick tree for frequencies"""
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, idx: int, val: int = 1) -> None:
        """add val at position idx (1-indexed)"""
        while idx <= self.n:
            self.bit[idx] += val
            idx += idx & -idx

    def sum(self, idx: int) -> int:
        """prefix sum [1..idx]"""
        res = 0
        while idx:
            res += self.bit[idx]
            idx -= idx & -idx
        return res


def main() -> None:
    input = sys.stdin.readline
    N = int(input())
    P = list(map(int, input().split()))

    bit = BIT(N)          # stores which numbers have already appeared
    ans = 0

    for i, x in enumerate(P, 1):               # i : current position (1-indexed)
        smaller = bit.sum(x - 1)               # numbers < x that are already on the left
        bigger = (i - 1) - smaller             # numbers > x that are on the left

        # x has to move left across `bigger` elements.
        # The edges it crosses cost: i-1, i-2, ..., i-bigger
        # Their sum is: bigger*i - bigger*(bigger+1)/2
        ans += bigger * i - bigger * (bigger + 1) // 2

        bit.add(x)                             # put x into the structure

    print(ans)


if __name__ == "__main__":
    main()