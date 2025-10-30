import sys

def main():
    N = int(sys.stdin.readline())
    D = [[0] * N for _ in range(N)]
    
    for i in range(N-1):
        line = list(map(int, sys.stdin.readline().split()))
        idx = 0
        for j in range(i+1, N):
            D[i][j] = line[idx]
            D[j][i] = line[idx]
            idx += 1
    
    size = 1 << N
    dp = [0] * size
    
    for mask in range(size):
        for i in range(N):
            if not (mask & (1 << i)):
                for j in range(i+1, N):
                    if not (mask & (1 << j)):
                        new_mask = mask | ( (1 << i) | (1 << j) )
                        if dp[new_mask] < dp[mask] + D[i][j]:
                            dp[new_mask] = dp[mask] + D[i][j]
    
    print(max(dp))

if __name__ == "__main__":
    main()