def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    D = [[0] * N for _ in range(N)]
    
    for i in range(N-1):
        for j in range(i+1, N):
            D[i][j] = int(data[index])
            D[j][i] = D[i][j]
            index += 1
    
    # Initialize dp array
    dp = [0] * (1 << N)
    
    for mask in range(1 << N):
        for i in range(N):
            if not (mask & (1 << i)):
                for j in range(i+1, N):
                    if not (mask & (1 << j)):
                        new_mask = mask | (1 << i) | (1 << j)
                        dp[new_mask] = max(dp[new_mask], dp[mask] + D[i][j])
    
    max_weight = max(dp)
    print(max_weight)

if __name__ == "__main__":
    main()