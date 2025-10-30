def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    D = [[0]*N for _ in range(N)]
    for i in range(N-1):
        line = list(map(int, input[ptr:ptr + (N - 1 - i)]))
        ptr += (N - 1 - i)
        for j in range(i+1, N):
            D[i][j] = line[j - (i+1)]
    dp = [-float('inf')] * (1 << N)
    dp[0] = 0
    for mask in range(1 << N):
        if dp[mask] == -float('inf'):
            continue
        available = []
        for i in range(N):
            if not (mask & (1 << i)):
                available.append(i)
        for i in range(len(available)):
            for j in range(i+1, len(available)):
                u = available[i]
                v = available[j]
                new_mask = mask | (1 << u) | (1 << v)
                if dp[new_mask] < dp[mask] + D[u][v]:
                    dp[new_mask] = dp[mask] + D[u][v]
    print(max(dp))

if __name__ == "__main__":
    main()