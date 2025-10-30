mod = 998244353

n, m = map(int, input().split())

a = pow(m-1, n, mod)

if n % 2 == 0:
    sign = 1
else:
    sign = -1

b = (sign * (m-1)) % mod

total = (a + b) % mod

print(total)