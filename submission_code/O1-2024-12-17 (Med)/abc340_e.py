import sys
sys.setrecursionlimit(10**7)

class FenwickRangePoint:
    """
    Fenwick (Binary Indexed) Tree for range update and point query.
    Uses two internal Fenwicks (bit1, bit2).

    All methods use 1-based indexing for positions.
    For an array of length n, valid indices are 1..n inclusive.
    """
    def __init__(self, n):
        self.n = n
        self.bit1 = [0]*(n+1)
        self.bit2 = [0]*(n+1)

    def _add(self, bit, idx, val):
        """ Add val to Fenwick 'bit' at position idx (1-based). """
        while idx <= self.n:
            bit[idx] += val
            idx += idx & -idx

    def range_update(self, l, r, val):
        """ Add 'val' to all positions in [l..r], 1-based. """
        if l > r:
            return
        # update BIT1
        self._add(self.bit1, l, val)
        if r+1 <= self.n:
            self._add(self.bit1, r+1, -val)
        # update BIT2
        self._add(self.bit2, l, val*(l-1))
        if r+1 <= self.n:
            self._add(self.bit2, r+1, -val*r)

    def _prefix_sum(self, idx):
        """ Returns the Fenwick prefix-sum (1-based) from 1..idx. """
        # sum(bit1) * idx - sum(bit2)
        s1, s2 = 0, 0
        i = idx
        while i > 0:
            s1 += self.bit1[i]
            i -= i & -i
        i = idx
        while i > 0:
            s2 += self.bit2[i]
            i -= i & -i
        return s1*idx - s2

    def point_query(self, idx):
        """ Get the value at position idx (1-based). """
        return self._prefix_sum(idx) - self._prefix_sum(idx-1)


def main():
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))

    # Special case: N=1, nothing really changes because everything loops to the same box.
    if N == 1:
        # The box always ends up with the same number of balls it started with.
        print(A[0])
        return

    # Build Fenwicks for range updates, point queries (size N, 1-based indexing).
    fenwicks = FenwickRangePoint(N)

    # Initialize Fenwicks with the initial counts A_i by doing range_update(i+1, i+1, A_i).
    for i in range(N):
        if A[i] != 0:
            fenwicks.range_update(i+1, i+1, A[i])

    # Process each operation
    for b in B:
        # 1) Determine how many balls are in box b
        #    (b is 0-based, Fenwicks is 1-based, so we query b+1)
        t = fenwicks.point_query(b+1)
        if t == 0:
            # No balls to move, skip
            continue
        # 2) Remove all t balls from box b
        fenwicks.range_update(b+1, b+1, -t)

        # 3) Distribute t among all boxes in the pattern
        q, r = divmod(t, N)
        # 3a) Add q to each of the N boxes if q > 0
        if q != 0:
            fenwicks.range_update(1, N, q)
        # 3b) Add 1 to r consecutive boxes, starting from (b+1) mod N
        if r > 0:
            start = (b+1) % N   # 0-based
            end = (b + r) % N  # 0-based
            if start <= end:
                # single segment [start..end]
                fenwicks.range_update(start+1, end+1, 1)
            else:
                # wrap around => two segments [start..N-1], [0..end]
                fenwicks.range_update(start+1, N, 1)
                fenwicks.range_update(1, end+1, 1)

    # After all M operations, compute final result
    # Just point-query each box
    ans = []
    for i in range(N):
        ans.append(str(fenwicks.point_query(i+1)))

    print(" ".join(ans))


# Do not forget to call main()
if __name__ == "__main__":
    main()