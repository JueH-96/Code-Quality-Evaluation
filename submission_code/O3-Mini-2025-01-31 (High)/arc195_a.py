def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Parse input
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    # Construct the leftmost matching positions L[i] for B[i] as a subsequence of A.
    L = [0] * M
    pos = 0
    for i in range(M):
        while pos < N and A[pos] != B[i]:
            pos += 1
        if pos == N:  # B is not a subsequence of A.
            sys.stdout.write("No")
            return
        L[i] = pos
        pos += 1  # Move to the next index for future matching

    # Construct the rightmost matching positions R[i] for B[i] by scanning from the end.
    R = [0] * M
    pos = N - 1
    for i in range(M - 1, -1, -1):
        while pos >= 0 and A[pos] != B[i]:
            pos -= 1
        if pos < 0:  # Should not happen as L found a valid subsequence.
            sys.stdout.write("No")
            return
        R[i] = pos
        pos -= 1

    # If there exists an index i for which L[i] < R[i],
    # then we can choose two different subsequences:
    # one using the leftmost path and one using the rightmost path.
    for i in range(M):
        if L[i] < R[i]:
            sys.stdout.write("Yes")
            return
    sys.stdout.write("No")

if __name__ == '__main__':
    main()