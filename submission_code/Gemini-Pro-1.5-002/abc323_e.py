# YOUR CODE HERE
def power(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def modInverse(n, p):
    return power(n, p - 2, p)

def solve():
    n, x = map(int, input().split())
    t = list(map(int, input().split()))
    
    mod = 998244353
    
    dp = [[0] * (x + 1) for _ in range(n + 1)]
    
    for i in range(n):
        dp[i][0] = modInverse(n, mod)

    for j in range(1, x + 1):
        for i in range(n):
            if j >= t[i]:
                dp[i][j] = sum(dp[k][j - t[i]] for k in range(n)) % mod
            else:
                dp[i][j] = 0
            dp[i][j] = (dp[i][j] * modInverse(n, mod)) % mod
            
    print(dp[0][x])

solve()