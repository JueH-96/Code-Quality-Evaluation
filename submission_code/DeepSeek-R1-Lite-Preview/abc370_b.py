def main():
    import sys
    lines = sys.stdin.read().splitlines()
    N = int(lines[0])
    A = [ [] for _ in range(N+1) ]
    for i in range(1, N+1):
        A[i] = list(map(int, lines[i].split()))
    current = 1
    for k in range(1, N+1):
        if current >= k:
            current = A[current][k-1]
        else:
            current = A[k][current-1]
    print(current)

if __name__ == '__main__':
    main()