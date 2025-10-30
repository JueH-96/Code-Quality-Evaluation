def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    X = list(map(int, input[ptr:ptr+M]))
    ptr += M
    A = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    # Pair each X_i with its A_i and sort by X_i
    paired = sorted(zip(X, A), key=lambda x: x[0])
    X_sorted = [x for x, a in paired]
    A_sorted = [a for x, a in paired]
    
    total = sum(A_sorted)
    if total != N:
        print(-1)
        return
    
    # Compute prefix sums
    prefix = [0] * (M + 1)
    for i in range(1, M+1):
        prefix[i] = prefix[i-1] + A_sorted[i-1]
    
    # Check conditions
    for i in range(M):
        if prefix[i+1] < X_sorted[i]:
            print(-1)
            return
    
    for i in range(M-1):
        if prefix[i+1] < (X_sorted[i+1] - 1):
            print(-1)
            return
    
    if prefix[M] < N:
        print(-1)
        return
    
    # Calculate minimal steps
    minimal_steps = 0
    for i in range(M):
        current_sum = prefix[i+1]
        current_X = X_sorted[i]
        excess = current_sum - current_X
        minimal_steps += excess * (N - current_X)
    
    print(minimal_steps)

if __name__ == '__main__':
    main()