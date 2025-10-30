import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    S = list(input().strip())
    # Convert to ints for faster ops
    for i in range(N):
        S[i] = 1 if S[i] == '1' else 0

    # We define A[i] = 1 if S[i] == S[i+1], for i in [0..N-2]
    # We need to support:
    #  - range sum on A
    #  - point updates on A
    class BIT:
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n+1)
        def add(self, i, v):
            # i: 0-based index
            i += 1
            while i <= self.n:
                self.fw[i] += v
                i += i & -i
        def sum(self, i):
            # sum [0..i], 0-based
            s = 0
            i += 1
            while i > 0:
                s += self.fw[i]
                i -= i & -i
            return s
        def range_sum(self, l, r):
            # sum over [l..r], 0-based
            if l > r:
                return 0
            return self.sum(r) - (self.sum(l-1) if l > 0 else 0)

    if N >= 2:
        bit = BIT(N-1)
        # build initial A into BIT
        for i in range(N-1):
            if S[i] == S[i+1]:
                bit.add(i, 1)
    else:
        bit = BIT(1)  # dummy for N=1

    out = []
    for _ in range(Q):
        t, L, R = map(int, input().split())
        L -= 1
        R -= 1
        if t == 1:
            # flip S[L..R]
            # Only A[L-1] and A[R] may change
            if L > 0:
                i = L-1
                old = 1 if S[i] == S[i+1] else 0
                # after flip, S[i+1] will be toggled, S[i] unchanged
                new = 1 if S[i] == (1 - S[i+1]) else 0
                if new != old:
                    bit.add(i, new - old)
            if R < N-1:
                i = R
                old = 1 if S[i] == S[i+1] else 0
                # after flip, S[i] will be toggled, S[i+1] unchanged
                new = 1 if (1 - S[i]) == S[i+1] else 0
                if new != old:
                    bit.add(i, new - old)
            # perform the flip
            for i in range(L, R+1):
                S[i] ^= 1
        else:
            # query if S[L..R] is good: no two consecutive equal
            if R - L <= 0:
                # single char always good
                out.append("Yes")
            else:
                cnt = bit.range_sum(L, R-1)
                out.append("Yes" if cnt == 0 else "No")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()