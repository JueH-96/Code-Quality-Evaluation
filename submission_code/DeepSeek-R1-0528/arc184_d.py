MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    points = []
    index = 1
    for i in range(n):
        x = int(data[index])
        y = int(data[index+1])
        index += 2
        points.append((x, y))
    
    points.sort(key=lambda p: p[0])
    a = [0] * (n+1)
    for i in range(1, n+1):
        a[i] = points[i-1][1]
    
    min_index = [[0] * (n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        min_idx = i
        for j in range(i, n+1):
            if a[j] < a[min_idx]:
                min_idx = j
            min_index[i][j] = min_idx
    
    dp = [[0] * (n+2) for _ in range(n+2)]
    for length in range(0, n+1):
        for l in range(1, n+1):
            r = l + length
            if r > n:
                continue
            if l > r:
                dp[l][r] = 1
            else:
                m = min_index[l][r]
                left_ways = dp[l][m-1] if l <= m-1 else 1
                right_ways = dp[m+1][r] if m+1 <= r else 1
                total = (left_ways * right_ways) % MOD
                total = (total + dp[l][r-1]) % MOD
                dp[l][r] = total
    print(dp[1][n] % MOD)

if __name__ == "__main__":
    main()