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
    A = list(map(int, input[ptr:ptr+N]))
    
    # Convert X to 0-based
    X = [x - 1 for x in X]
    
    if K == 0:
        print(' '.join(map(str, A)))
        return
    
    max_level = 60  # Since 2^60 > 1e18
    # Initialize lift table
    lift = [[0] * max_level for _ in range(N)]
    for i in range(N):
        lift[i][0] = X[i]
    
    for j in range(1, max_level):
        for i in range(N):
            lift[i][j] = lift[lift[i][j-1]][j-1]
    
    result = []
    for i in range(N):
        current = i
        k = K
        for j in range(max_level - 1, -1, -1):
            if (k >> j) & 1:
                current = lift[current][j]
        result.append(A[current])
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()