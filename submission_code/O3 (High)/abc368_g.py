import sys
from typing import List, Tuple


# ------------------  Lines & Hull  ------------------ #
Line = Tuple[int, int]          # (slope , intercept)


def build_hull(raw: List[Line]) -> List[Line]:
    """
    From a list of lines return the upper envelope (for x >= 0)
    keeping slopes in strictly increasing order.
    Each line is y = m * x + b.
    """
    # keep only the largest intercept for each slope
    best = {}
    for m, b in raw:
        if (m not in best) or (b > best[m]):
            best[m] = b

    items = sorted(best.items())          # sort by slope (ascending)

    hull: List[Line] = []
    for m, b in items:
        # usual convex-hull pruner
        while len(hull) >= 2 and _is_bad(hull[-2], hull[-1], (m, b)):
            hull.pop()
        hull.append((m, b))
    return hull


def _is_bad(l1: Line, l2: Line, l3: Line) -> bool:
    """
    Return True iff line l2 is never on the upper envelope
    formed by l1, l2, l3 (all with strictly increasing slopes).
         (c1 - c2)/(m2 - m1) >= (c2 - c3)/(m3 - m2)
    cross multiplied to avoid floats.
    """
    m1, c1 = l1
    m2, c2 = l2
    m3, c3 = l3
    return (c1 - c2) * (m3 - m2) >= (c2 - c3) * (m2 - m1)


def eval_lines(lines: List[Line], x: int) -> int:
    """
    Evaluate max(m*x+b) for the given x (x fits in 64 bits,
    but m and b may be huge).
    """
    best = -1
    for m, b in lines:
        val = m * x + b
        if val > best:
            best = val
    return best


# ------------------  Segment Tree  ------------------ #
class SegmentTree:
    def __init__(self, A: List[int], B: List[int]):
        self.n = len(A)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.data: List[List[Line]] = [None] * (2 * self.size)

        # build leaves
        for i in range(self.size):
            if i < self.n:
                self.data[self.size + i] = self._leaf(A[i], B[i])
            else:
                self.data[self.size + i] = [(1, 0)]          # identity

        # build internal nodes
        for idx in range(self.size - 1, 0, -1):
            self.data[idx] = self._merge(self.data[idx << 1], self.data[idx << 1 | 1])

    @staticmethod
    def _leaf(a: int, b: int) -> List[Line]:
        # single element function f(v) = max(v + a, v * b)
        raw = [(1, a), (b, 0)]
        return build_hull(raw)

    @staticmethod
    def _merge(left: List[Line], right: List[Line]) -> List[Line]:
        """
        Compose two segments:
            first apply 'left', then 'right'
            (i.e. g(v) = right( left(v) ))
        Result is its upper envelope.
        """
        raw: List[Line] = []
        for mr, cr in right:
            for ml, cl in left:
                raw.append((mr * ml, mr * cl + cr))
        return build_hull(raw)

    # -----------  point updates  ----------- #
    def update_A(self, idx: int, newA: int, B: List[int]):
        pos = idx + self.size
        # update leaf
        self.data[pos] = self._leaf(newA, B[idx])

        # propagate upwards
        pos >>= 1
        while pos:
            self.data[pos] = self._merge(self.data[pos << 1], self.data[pos << 1 | 1])
            pos >>= 1

    def update_B(self, idx: int, newB: int, A: List[int]):
        pos = idx + self.size
        self.data[pos] = self._leaf(A[idx], newB)
        pos >>= 1
        while pos:
            self.data[pos] = self._merge(self.data[pos << 1], self.data[pos << 1 | 1])
            pos >>= 1

    # -----------  range query  ----------- #
    def query(self, l: int, r: int) -> int:
        """
        return answer for interval [l, r] (0-based inclusive)
        """
        l += self.size
        r += self.size + 1       # make r exclusive

        left_parts: List[List[Line]] = []
        right_parts: List[List[Line]] = []

        while l < r:
            if l & 1:
                left_parts.append(self.data[l])
                l += 1
            if r & 1:
                r -= 1
                right_parts.append(self.data[r])
            l >>= 1
            r >>= 1

        v = 0
        for node in left_parts:
            v = eval_lines(node, v)
        for node in reversed(right_parts):
            v = eval_lines(node, v)
        return v


# ------------------  Main Routine  ------------------ #
def main() -> None:
    input_data = sys.stdin.read().strip().split()
    it = iter(input_data)

    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]

    seg = SegmentTree(A, B)

    Q = int(next(it))
    out_lines = []

    for _ in range(Q):
        tp = int(next(it))
        if tp == 1:
            i = int(next(it)) - 1
            x = int(next(it))
            A[i] = x
            seg.update_A(i, x, B)
        elif tp == 2:
            i = int(next(it)) - 1
            x = int(next(it))
            B[i] = x
            seg.update_B(i, x, A)
        else:
            l = int(next(it)) - 1
            r = int(next(it)) - 1
            ans = seg.query(l, r)
            out_lines.append(str(ans))

    sys.stdout.write('
'.join(out_lines))


if __name__ == "__main__":
    main()