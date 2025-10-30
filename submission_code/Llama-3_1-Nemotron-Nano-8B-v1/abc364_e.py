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
        idx += 1
        B = int(input[idx])
        idx += 1
        dishes.append((A, B))
    
    # Initialize DP table. dp[a][b] is the maximum number of dishes with sumA = a and sumB = b.
    dp = [[-1] * (Y + 1) for _ in range(X + 1)]
    dp[0][0] = 0
    
    for a, b in dishes:
        # Iterate in reverse to avoid reusing the same dish multiple times
        for current_a in range(X, a - 1, -1):
            for current_b in range(Y, b - 1, -1):
                if dp[current_a - a][current_b - b] != -1:
                    if dp[current_a][current_b] < dp[current_a - a][current_b - b] + 1:
                        dp[current_a][current_b] = dp[current_a - a][current_b - b] + 1
    
    max_dishes = 0
    for a in range(X + 1):
        for b in range(Y + 1):
            if dp[a][b] > max_dishes:
                max_dishes = dp[a][b]
    
    print(max_dishes)

if __name__ == "__main__":
    main()