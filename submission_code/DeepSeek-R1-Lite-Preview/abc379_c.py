def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    X = [int(data[i + 2]) for i in range(M)]
    A = [int(data[i + 2 + M]) for i in range(M)]
    
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return
    
    # Sort the positions
    positions = sorted(zip(X, A), key=lambda x: x[0])
    X_sorted = [pos[0] for pos in positions]
    A_sorted = [pos[1] for pos in positions]
    
    operations = 0
    cumulative_excess = 0
    
    for i in range(M):
        if i > 0:
            prev_X = X_sorted[i-1]
            current_X = X_sorted[i]
            gap = current_X - prev_X - 1
            if cumulative_excess < gap:
                print(-1)
                return
            operations += gap * (gap + 1) // 2
            cumulative_excess -= gap
        # At position X_sorted[i]
        cumulative_excess += A_sorted[i] - 1
    
    # Check remaining cells after the last position
    if N > X_sorted[-1]:
        gap = N - X_sorted[-1]
        if cumulative_excess < gap:
            print(-1)
            return
        operations += gap * (gap + 1) // 2
        cumulative_excess -= gap
    
    if cumulative_excess == 0:
        print(operations)
    else:
        print(-1)

if __name__ == '__main__':
    main()