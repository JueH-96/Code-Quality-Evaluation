def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    M = int(input[idx])
    idx +=1
    X = list(map(int, input[idx:idx+M]))
    idx +=M
    A = list(map(int, input[idx:idx+M]))
    idx +=M
    
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return
    
    # Sort the cells by their positions
    sorted_pairs = sorted(zip(X, A))
    current_sum = 0
    total_moves = 0
    possible = True
    
    for x, a in sorted_pairs:
        s = current_sum + 1
        if x > s:
            possible = False
            break
        # Calculate moves for this group
        total_moves += a * (s - x) + a * (a - 1) // 2
        current_sum += a
    
    if possible:
        print(total_moves)
    else:
        print(-1)

if __name__ == "__main__":
    main()