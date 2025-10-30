MOD = 998244353

n = int(input())
a = list(map(int, input().split()))

result = 1
for i in range(n):
    result = result * (i + 1 - a[i]) % MOD

print(result)