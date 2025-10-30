def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N
    X = [x - 1 for x in X]
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Initialize binary lifting table
    LOG = 60
    lift = [[0] * N for _ in range(LOG)]
    lift[0] = X.copy()

    for j in range(1, LOG):
        for i in range(N):
            lift[j][i] = lift[j-1][lift[j-1][i]]

    # Compute the result for each position
    result = []
    for i in range(N):
        current = i
        for j in range(LOG):
            if (K >> j) & 1:
                current = lift[j][current]
        result.append(A[current])

    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()