import sys

# ------------------------------------------------------------
# Fenwick Tree (Binary Indexed Tree)
# ------------------------------------------------------------

class Fenwick:
    """1-indexed Fenwick that supports add / prefix-sum"""
    __slots__ = ('n', 'bit')
    def __init__(self, n: int):
        self.n  = n
        self.bit = [0]*(n+1)

    def add(self, idx: int, delta: int) -> None:
        """point add: a[idx] += delta   (1 <= idx <= n)"""
        while idx <= self.n:
            self.bit[idx] += delta
            idx += idx & -idx

    def sum(self, idx: int) -> int:
        """prefix sum: a[1] + ... + a[idx]   (0 <= idx <= n)"""
        s = 0
        while idx:
            s += self.bit[idx]
            idx -= idx & -idx
        return s

    def range_sum(self, l: int, r: int) -> int:
        """sum on [l, r] (1-indexed, empty when l>r)"""
        if l > r:
            return 0
        return self.sum(r) - self.sum(l-1)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N  = int(next(it))
    Q  = int(next(it))
    S  = list(map(int, next(it).decode()))
    
    # Fenwick for flips: range add / point query (we store differences)
    flipBIT = Fenwick(N+2)          # extra space for r+1 update
    
    # Fenwick for equality array A[i] = 1 if S[i]==S[i+1] (1 <= i <= N-1)
    equalBIT = Fenwick(N)           # size N is safe (index N unused)
    for i in range(1, N):           # build initial equality BIT
        if S[i-1] == S[i]:
            equalBIT.add(i, 1)

    out_lines = []

    # helper to fetch current bit at position idx (1-indexed)
    def current_bit(idx: int) -> int:
        return S[idx-1] ^ (flipBIT.sum(idx) & 1)

    for _ in range(Q):
        t = int(next(it))
        L = int(next(it))
        R = int(next(it))

        if t == 1:                        # flip query
            # -------- handle boundary L-1 ------------
            if L > 1:
                # old equality before flip
                b1   = current_bit(L-1)
                b2   = current_bit(L)
                old  = 1 if b1 == b2 else 0
            if R < N:
                b3   = current_bit(R)
                b4   = current_bit(R+1)
                oldR = 1 if b3 == b4 else 0

            # apply range flip on [L, R]
            flipBIT.add(L, 1)
            flipBIT.add(R+1, -1)

            # -------- recompute equality and update BIT ----------
            if L > 1:
                nb2  = current_bit(L)      # after flip
                new  = 1 if b1 == nb2 else 0
                equalBIT.add(L-1, new - old)

            if R < N:
                nb3  = current_bit(R)      # after flip
                newR = 1 if nb3 == b4 else 0
                equalBIT.add(R, newR - oldR)

        else:                              # type 2 query
            # substring is good <=> no equal pair inside
            if L == R:
                out_lines.append("Yes")
            else:
                total = equalBIT.range_sum(L, R-1)
                out_lines.append("Yes" if total == 0 else "No")

    sys.stdout.write("
".join(out_lines))

# ------------------------------------------------------------
# execute
# ------------------------------------------------------------
if __name__ == "__main__":
    main()