import math

def solve():
    n = int(input())
    p = list(map(int, input().split()))
    dp = [[-float('inf')] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][1] = p[i-1]
    
    for j in range(2, n + 1):
        current_max = -float('inf')
        for i in range(j, n + 1):
            if i == j:
                current_max = dp[j-1][j-1]
            else:
                current_max = max(current_max, dp[i-1][j-1])
            dp[i][j] = p[i-1] + 0.9 * current_max
            
    max_rating = -float('inf')
    for k in range(1, n + 1):
        max_n_k = -float('inf')
        for i in range(k, n + 1):
            max_n_k = max(max_n_k, dp[i][k])
        if max_n_k != -float('inf'):
            d_k = 0
            for i_val in range(1, k + 1):
                d_k += (0.9)**(k - i_val)
            if d_k == 0:
                continue
            r_k = max_n_k / d_k - 1200 / math.sqrt(k)
            max_rating = max(max_rating, r_k)
            
    print(f"{max_rating:.17f}")

if __name__ == '__main__':
    solve()