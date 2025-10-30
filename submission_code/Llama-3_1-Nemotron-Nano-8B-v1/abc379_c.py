def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    X = []
    A = []
    for _ in range(M):
        X.append(int(input[idx]))
        idx += 1
        A.append(int(input[idx]))
        idx += 1
    
    if sum(A) != N:
        print(-1)
        return
    
    stones = sorted(zip(X, A), key=lambda x: x[0])
    
    current_pos = 1
    total_moves = 0
    
    for x, a in stones:
        start = max(x, current_pos)
        end = start + a - 1
        if end > N:
            print(-1)
            return
        if start > current_pos:
            print(-1)
            return
        # Calculate the sum of (start - x) to (end - x)
        sum_moves = (start + end - 2 * x) * a // 2
        total_moves += sum_moves
        current_pos = end + 1
    
    if current_pos != N + 1:
        print(-1)
    else:
        print(total_moves)

if __name__ == "__main__":
    main()