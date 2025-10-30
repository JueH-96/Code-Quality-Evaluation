import sys

class BIT:
    """
    Fenwick tree (Binary Indexed Tree) that supports
    add(idx, val) and prefix_sum(idx) in O(log n).
    Indices given to the public methods are 0-based,
    internally the tree is 1-based.
    """
    __slots__ = ("n", "tree")

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, idx: int, val: int) -> None:
        idx += 1  # to 1-based
        while idx <= self.n:
            self.tree[idx] += val
            idx += idx & -idx

    def prefix_sum(self, idx: int) -> int:
        """sum of [0 .. idx] (idx inclusive)."""
        idx += 1  # to 1-based
        res = 0
        while idx:
            res += self.tree[idx]
            idx -= idx & -idx
        return res


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:             # empty input guard (should never occur)
        return
    N, M = data[0], data[1]
    A = data[2:]

    bit = BIT(M)             # stores counts of prefix sums modulo M
    prefix = 0               # S_0
    bit.add(prefix, 1)       # insert S_0
    total_prefix_sum = 0     # sum of all previous prefix values
    count_prefix = 1         # number of previous prefixes (currently only S_0)

    answer = 0

    for a in A:
        prefix = (prefix + a) % M

        # number of previous prefixes with value <= current prefix
        le_cnt = bit.prefix_sum(prefix)
        gt_cnt = count_prefix - le_cnt

        # contribution of all subarrays ending at current position
        # Formula: current_prefix * count_prefix - total_prefix_sum + gt_cnt * M
        contribution = prefix * count_prefix - total_prefix_sum + gt_cnt * M
        answer += contribution

        # update structures with current prefix
        bit.add(prefix, 1)
        total_prefix_sum += prefix
        count_prefix += 1

    print(answer)


if __name__ == "__main__":
    main()