def main():
    import sys
    data = sys.stdin.read().split()
    index = 0
    N = int(data[index])
    index += 1
    D = [[0] * N for _ in range(N)]
    for i in range(N-1):
        for j in range(i+1, N):
            D[i][j] = int(data[index])
            D[j][i] = D[i][j]
            index += 1
    size = 1 << N
    dp = [float('-inf')] * size
    dp[0] = 0
    for mask in range(size):
        if dp[mask] == float('-inf'):
            continue
        # Find the smallest unused vertex
        i = 0
        while i < N and (mask & (1 << i)):
            i += 1
        if i == N:
            continue
        # Try pairing i with any j > i that is not used
        for j in range(i+1, N):
            if not (mask & (1 << j)):
                new_mask = mask | (1 << i) | (1 << j)
                dp[new_mask] = max(dp[new_mask], dp[mask] + D[i][j])
    # Find the maximum dp[mask]
    max_val = 0
    for mask in range(size):
        if dp[mask] > max_val:
            max_val = dp[mask]
    print(max_val)

if __name__ == "__main__":
    main()