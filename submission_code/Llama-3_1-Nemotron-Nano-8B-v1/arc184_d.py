MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    data = list(map(int, input[1:]))
    points = []
    for i in range(n):
        x = data[2*i]
        y = data[2*i + 1]
        points.append((x, y))
    points.sort()
    y_list = [y for x, y in points]
    y_perm = [i + 1 for i in range(n)]
    y_list = [y_perm[p] for p in points]
    
    min_y = [float('inf')] * (n + 1)
    dp = [0] * (n + 2)
    dp[0] = 1
    min_y[0] = float('inf')
    
    for i in range(n):
        x_i, y_i = points[i]
        y = y_list[i]
        new_dp = [0] * (n + 2)
        new_min_y = [float('inf')] * (n + 2)
        
        for j in range(n + 1):
            if dp[j] == 0:
                continue
            current_min = min_y[j]
            new_min = min(current_min, y)
            new_dp[j] = (new_dp[j] + dp[j]) % MOD
            new_min_y[j] = new_min
            
            if current_min < y:
                new_dp[j] = (new_dp[j] + dp[j]) % MOD
                new_min_y[j] = current_min
        
        dp, new_dp = new_dp, [0] * (n + 2)
        min_y = new_min_y
    
    total = 0
    for val in dp:
        total = (total + val) % MOD
    print(total)

if __name__ == "__main__":
    main()