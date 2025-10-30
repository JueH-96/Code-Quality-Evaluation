def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    X = list(map(int, input[idx:idx+M]))
    idx += M
    A = list(map(int, input[idx:idx+M]))
    idx += M
    
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return
    
    # Combine and sort by X
    combined = sorted(zip(X, A), key=lambda x: x[0])
    X_sorted = [x for x, a in combined]
    A_sorted = [a for x, a in combined]
    
    current = 1
    total_ops = 0
    possible = True
    
    for x, a in zip(X_sorted, A_sorted):
        start = max(x, current)
        end = start + a - 1
        if end > N:
            possible = False
            break
        # Calculate operations for this group
        total_ops += a * (start - x)
        total_ops += a * (a - 1) // 2
        current = end + 1
    
    if not possible:
        print(-1)
    else:
        print(total_ops)

if __name__ == "__main__":
    main()