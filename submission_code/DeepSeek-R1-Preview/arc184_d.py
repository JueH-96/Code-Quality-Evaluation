MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    points = []
    idx = 1
    for _ in range(n):
        x = int(data[idx])
        y = int(data[idx+1])
        points.append((x, y))
        idx += 2
    
    points.sort()
    y_coords = [y for x, y in points]
    
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for i in range(n):
        new_dp = [0] * (n + 1)
        current_y = y_coords[i]
        for length in range(n + 1):
            if dp[length] == 0:
                continue
            new_dp[length] = (new_dp[length] + dp[length]) % MOD
            if length == 0 or current_y < y_coords[i - length]:
                new_dp[length + 1] = (new_dp[length + 1] + dp[length]) % MOD
        dp = new_dp
    
    total = sum(dp) % MOD
    print(total)

if __name__ == '__main__':
    main()