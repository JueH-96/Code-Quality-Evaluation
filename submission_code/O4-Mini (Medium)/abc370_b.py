def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # Read the triangular matrix A where A[i][j] is defined for 1 <= j <= i <= N
    # We'll store it 1-indexed.
    A = [[0] * (i + 1) for i in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, i + 1):
            A[i][j] = int(next(it))
    # Start with current element = 1
    cur = 1
    # Combine with elements 1 through N in order
    for k in range(1, N + 1):
        if cur >= k:
            cur = A[cur][k]
        else:
            cur = A[k][cur]
    # Output the final element
    print(cur)

if __name__ == "__main__":
    main()