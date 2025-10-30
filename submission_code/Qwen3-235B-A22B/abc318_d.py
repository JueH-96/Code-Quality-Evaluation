import sys

def main():
    n = int(sys.stdin.readline())
    data = []
    for i in range(n-1):
        data += list(map(int, sys.stdin.readline().split()))
    d = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            d[i][j] = data[idx]
            d[j][i] = d[i][j]
            idx += 1
    dp = [-float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        if dp[mask] == -float('inf'):
            continue
        for i in range(n):
            if (mask & (1 << i)):
                continue
            for j in range(i + 1, n):
                if (mask & (1 << j)):
                    continue
                new_mask = mask | (1 << i) | (1 << j)
                if dp[new_mask] < dp[mask] + d[i][j]:
                    dp[new_mask] = dp[mask] + d[i][j]
    print(int(max(dp)))

if __name__ == '__main__':
    main()