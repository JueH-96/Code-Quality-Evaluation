# YOUR CODE HERE
def power(a, b):
    res = 1
    a %= 998244353
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % 998244353
        a = (a * a) % 998244353
        b //= 2
    return res

def inverse(n):
    return power(n, 998244353 - 2)

n = int(input())
a = list(map(int, input().split()))

dp = [[0] * 11 for _ in range(n + 1)]
dp[0][0] = 1

total_outcomes = 1
for i in range(1, n + 1):
    total_outcomes = (total_outcomes * a[i-1]) % 998244353
    for j in range(11):
        for k in range(1, min(j + 1, a[i-1] + 1)):
            dp[i][j] = (dp[i][j] + dp[i-1][j-k]) % 998244353
        dp[i][j] = (dp[i][j] + dp[i-1][j] * max(0, a[i-1] - max(0, j - (10 - j)))) % 998244353


favorable_outcomes = 0
for i in range(1, n + 1):
    temp = 0
    for j in range(1, 11):
        temp = (temp + dp[i][j] * power(a[i-1], n - 1)) % 998244353
    favorable_outcomes = (favorable_outcomes + temp * inverse(total_outcomes)) % 998244353
    
favorable_outcomes = (favorable_outcomes * total_outcomes) % 998244353

print((favorable_outcomes * inverse(total_outcomes)) % 998244353)