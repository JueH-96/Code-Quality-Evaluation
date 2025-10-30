def main():
    import sys
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    A = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, i + 1):
            A[i][j] = int(data[ptr])
            ptr += 1
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            A[i][j] = A[j][i]
    c = 1
    for e in range(1, N + 1):
        c = A[c][e]
    print(c)

if __name__ == '__main__':
    main()