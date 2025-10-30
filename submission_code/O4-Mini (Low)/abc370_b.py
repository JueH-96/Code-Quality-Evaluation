def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # Prepare an (N+1)x(N+1) matrix, 1-based
    A = [[0] * (N+1) for _ in range(N+1)]
    # Read the lower-triangular entries
    for i in range(1, N+1):
        for j in range(1, i+1):
            A[i][j] = int(next(it))
    # Start combining from element 1
    current = 1
    for x in range(1, N+1):
        if current >= x:
            current = A[current][x]
        else:
            current = A[x][current]
    # Output the result
    print(current)

if __name__ == "__main__":
    main()