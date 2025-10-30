def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    Y = int(input[idx])
    idx += 1
    dishes = []
    for _ in range(N):
        A = int(input[idx])
        B = int(input[idx + 1])
        dishes.append((A, B))
        idx += 2
    
    INF = float('-inf')
    max_X = X
    max_Y = Y
    dp = [[INF] * (max_Y + 1) for _ in range(max_X + 1)]
    dp[0][0] = 0
    max_ans = 0
    
    for A, B in dishes:
        # Iterate backwards to prevent using the same dish multiple times
        for a in range(max_X, -1, -1):
            for b in range(max_Y, -1, -1):
                if dp[a][b] != INF:
                    new_a = a + A
                    new_b = b + B
                    if new_a <= max_X and new_b <= max_Y:
                        if dp[new_a][new_b] < dp[a][b] + 1:
                            dp[new_a][new_b] = dp[a][b] + 1
                    else:
                        # Check if this new count is better than current max_ans
                        current_count = dp[a][b] + 1
                        if current_count > max_ans:
                            max_ans = current_count
    
    # Find the maximum valid count in dp
    max_valid = 0
    for a in range(max_X + 1):
        for b in range(max_Y + 1):
            if dp[a][b] > max_valid:
                max_valid = dp[a][b]
    
    print(max(max_valid, max_ans))

if __name__ == "__main__":
    main()