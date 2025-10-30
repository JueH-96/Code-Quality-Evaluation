def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    
    dishes = []
    index = 3
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        dishes.append((A, B))
        index += 2
    
    # Initialize DP table
    dp = [[0] * (Y + 1) for _ in range(X + 1)]
    
    for a_i, b_i in dishes:
        # Iterate in reverse to avoid reprocessing the same dish
        for a in range(X, -1, -1):
            for b in range(Y, -1, -1):
                if a + a_i <= X and b + b_i <= Y:
                    if dp[a][b] + 1 > dp[a + a_i][b + b_i]:
                        dp[a + a_i][b + b_i] = dp[a][b] + 1
    
    max_val = 0
    for row in dp:
        current_max = max(row)
        if current_max > max_val:
            max_val = current_max
    
    print(max_val)

if __name__ == '__main__':
    main()