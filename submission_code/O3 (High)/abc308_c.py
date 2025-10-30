import sys
from functools import cmp_to_key

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))

    # store (heads, total_tosses, person_id)
    people = []
    for idx in range(1, n + 1):
        a = int(next(it))
        b = int(next(it))
        people.append((a, a + b, idx))

    # comparator that orders by success-rate descending, then id ascending
    def cmp(x, y):
        ax, sx, ix = x   # ax / sx  is the success rate of x
        ay, sy, iy = y   # ay / sy  is the success rate of y

        # compare ax/sx and ay/sy without using floating point
        lhs = ax * sy
        rhs = ay * sx
        if lhs > rhs:          # x has higher success rate
            return -1
        if lhs < rhs:          # y has higher success rate
            return 1
        # equal success rate: smaller id first
        return -1 if ix < iy else 1 if ix > iy else 0

    people.sort(key=cmp_to_key(cmp))
    print(' '.join(str(p[2]) for p in people))

if __name__ == "__main__":
    main()