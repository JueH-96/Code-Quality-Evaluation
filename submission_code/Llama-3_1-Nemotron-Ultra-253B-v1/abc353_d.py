n = int(input())
a = list(map(int, input().split()))
mod = 998244353

sum1 = 0
prefix = 0
for num in a:
    d = len(str(num))
    pow10 = pow(10, d, mod)
    sum1 = (sum1 + prefix * pow10) % mod
    prefix = (prefix + num) % mod

sum2 = 0
for j in range(n):
    sum2 = (sum2 + j * a[j]) % mod

total = (sum1 + sum2) % mod
print(total)