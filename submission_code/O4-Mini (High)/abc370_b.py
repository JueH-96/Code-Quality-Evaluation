def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # A[i][j] is defined for i >= j
    A = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, i + 1):
            A[i][j] = int(next(it))
    cur = 1
    # Combine cur with x for x = 1..N
    for x in range(1, N + 1):
        if cur >= x:
            cur = A[cur][x]
        else:
            cur = A[x][cur]
    sys.stdout.write(str(cur))

if __name__ == "__main__":
    main()