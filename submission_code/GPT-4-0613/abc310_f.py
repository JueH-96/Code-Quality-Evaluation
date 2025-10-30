MOD = 998244353
MAX = 10**6+10
inv = [0]*MAX
inv[1] = 1
for i in range(2, MAX):
    inv[i] = inv[MOD%i]*(MOD-MOD//i)%MOD

N = int(input())
A = list(map(int, input().split()))
dp = [0]*11
dp[0] = 1
for a in A:
    ndp = [0]*11
    for i in range(11):
        if dp[i] == 0: continue
        for j in range(min(i, a+1)):
            ndp[i-j] = (ndp[i-j] + dp[i]*inv[a+1])%MOD
    dp = ndp
print((pow(sum(A), 10, MOD) - sum(dp) + dp[10])%MOD)