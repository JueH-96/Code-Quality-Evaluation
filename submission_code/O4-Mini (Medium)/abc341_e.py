import sys
import threading

def main():
    import sys
    data = sys.stdin
    # Fast reader
    line = data.readline().split()
    N = int(line[0]); Q = int(line[1])
    S = data.readline().strip()
    # 1-based indexing for S
    s = [''] * (N + 2)
    for i, ch in enumerate(S, start=1):
        s[i] = ch

    # Build A array: A[i] = 1 if s[i] == s[i+1], else 0, for i=1..N-1
    # We'll build a Fenwick Tree (BIT) over A[1..N-1].
    size = max(1, N - 1)
    BIT = [0] * (size + 1)
    A = [0] * (N + 2)  # store current A-values, 1-based

    def bit_add(i, v):
        # add v at index i in BIT
        while i <= size:
            BIT[i] += v
            i += i & -i

    def bit_sum(i):
        # prefix sum 1..i
        s = 0
        while i > 0:
            s += BIT[i]
            i -= i & -i
        return s

    def bit_range_sum(l, r):
        # sum A[l..r]
        return bit_sum(r) - bit_sum(l - 1)

    # initialize
    for i in range(1, N):
        if s[i] == s[i+1]:
            A[i] = 1
            bit_add(i, 1)

    out = []
    for _ in range(Q):
        parts = data.readline().split()
        typ = parts[0]
        L = int(parts[1]); R = int(parts[2])
        if typ == '1':
            # flip s[L..R], so A[L-1] and A[R] toggle if in range
            if L > 1:
                i = L - 1
                # new A[i] = 1 - A[i]
                if A[i] == 0:
                    A[i] = 1
                    bit_add(i, 1)
                else:
                    A[i] = 0
                    bit_add(i, -1)
            if R < N:
                i = R
                if A[i] == 0:
                    A[i] = 1
                    bit_add(i, 1)
                else:
                    A[i] = 0
                    bit_add(i, -1)
        else:
            # query type 2
            if L == R:
                out.append("Yes")
            else:
                # check sum A[L..R-1] == 0
                if bit_range_sum(L, R-1) == 0:
                    out.append("Yes")
                else:
                    out.append("No")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()