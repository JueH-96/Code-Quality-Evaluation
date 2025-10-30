MOD = 998244353
K = int(input().strip())
C = list(map(int, input().split()))
 
fact = [1]*(K+1)
inv = [1]*(K+1)
for i in range(1, K+1):
    fact[i] = fact[i-1]*i%MOD
    inv[i] = pow(fact[i], MOD-2, MOD)
 
dp = [0]*(K+1)
dp[0] = 1
for c in C:
    dp2 = [0]*(K+1)
    for i in range(K+1):
        for j in range(min(i, c)+1):
            dp2[i] = (dp2[i]+dp[i-j]*fact[i]*inv[j]%MOD*inv[i-j]%MOD)%MOD
    dp = dp2
print(dp[-1])