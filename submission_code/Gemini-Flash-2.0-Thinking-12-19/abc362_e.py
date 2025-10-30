def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    counts = [0] * (n + 1)
    counts[1] = n
    
    dp = [None] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = [None] * (n + 1)
        for k in range(2, n + 1):
            dp[i][k] = {}
            
    for i in range(2, n + 1):
        for j in range(1, i):
            diff = a[i-1] - a[j-1]
            if diff not in dp[i][2]:
                dp[i][2][diff] = 0
            dp[i][2][diff] += 1
            
    for k in range(3, n + 1):
        for i in range(1, n + 1):
            for j in range(1, i):
                diff = a[i-1] - a[j-1]
                if diff in dp[j][k-1]:
                    if diff not in dp[i][k]:
                        dp[i][k][diff] = 0
                    dp[i][k][diff] += dp[j][k-1][diff]
                    
    mod = 998244353
    
    for k in range(2, n + 1):
        count_k = 0
        for i in range(1, n + 1):
            for diff in dp[i][k]:
                count_k = (count_k + dp[i][k][diff]) % mod
        counts[k] = count_k
        
    result = []
    for k in range(1, n + 1):
        result.append(str(counts[k]))
        
    print(" ".join(result))

if __name__ == '__main__':
    solve()