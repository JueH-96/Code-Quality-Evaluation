n, k = map(int, input().split())
a = list(map(int, input().split()))

MOD = 998244353

total = 0
for l in range(n):
    current_sum = 0
    for r in range(l, n):
        current_sum = (current_sum + a[r]) % MOD
        total = (total + pow(current_sum, k, MOD)) % MOD

print(total)