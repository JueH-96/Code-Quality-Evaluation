def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    X = []
    A = []
    for i in range(M):
        x = int(data[2 + i])
        X.append(x)
    for i in range(M):
        a = int(data[2 + M + i])
        A.append(a)
    
    if sum(A) != N:
        print(-1)
        return
    
    # Sort X and corresponding A
    sorted_XA = sorted(zip(X, A))
    X = [x for x, a in sorted_XA]
    A = [a for x, a in sorted_XA]
    
    cumulative_stones = 0
    prev_X = 0
    index = 0
    for x, a in zip(X, A):
        for k in range(prev_X + 1, x):
            if cumulative_stones < k:
                print(-1)
                return
        cumulative_stones += a
        prev_X = x
    
    # Check remaining cells from prev_X to N
    for k in range(prev_X + 1, N + 1):
        if cumulative_stones < k:
            print(-1)
            return
    
    # Calculate operations
    sum_j = N * (N + 1) // 2
    sum_X_A = sum(x * a for x, a in zip(X, A))
    operations = sum_j - sum_X_A
    print(operations)

if __name__ == "__main__":
    main()