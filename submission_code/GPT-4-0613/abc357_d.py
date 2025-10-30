N = int(input().strip())
MOD = 998244353
ten = [1]
for _ in range(60):
    ten.append(ten[-1]*10%MOD)
ans = 0
for i in range(60):
    if ((N>>i)&1):
        ans = (ans + N*ten[i]%MOD) % MOD
print(ans)