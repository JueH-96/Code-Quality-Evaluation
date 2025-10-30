def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    # Read input sizes and sequences
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = int(next(it))
    B = [int(next(it)) for _ in range(M)]
    L = int(next(it))
    C = [int(next(it)) for _ in range(L)]
    Q = int(next(it))
    X = [int(next(it)) for _ in range(Q)]

    # Compute all pairwise sums of A and B
    S = [a + b for a in A for b in B]
    # Compute all triple sums by adding each c in C
    # Use a set for O(1) membership tests
    T = {s + c for s in S for c in C}

    out = []
    # Answer queries
    for x in X:
        out.append("Yes" if x in T else "No")

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()