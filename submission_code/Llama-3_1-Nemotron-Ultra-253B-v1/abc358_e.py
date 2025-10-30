mod = 998244353

max_n = 1000
fact = [1] * (max_n + 1)
for i in range(1, max_n + 1):
    fact[i] = fact[i-1] * i % mod

inv_fact = [1] * (max_n + 1)
inv_fact[max_n] = pow(fact[max_n], mod - 2, mod)
for i in range(max_n - 1, -1, -1):
    inv_fact[i] = inv_fact[i+1] * (i+1) % mod

K = int(input())
C = list(map(int, input().split()))

product_poly = [0] * (K + 1)
product_poly[0] = 1

for ci in C:
    new_poly = [0] * (K + 1)
    for a in range(K + 1):
        if product_poly[a] == 0:
            continue
        max_b = min(ci, K - a)
        for b in range(0, max_b + 1):
            new_poly[a + b] = (new_poly[a + b] + product_poly[a] * inv_fact[b]) % mod
    product_poly = new_poly

ans = 0
for l in range(1, K + 1):
    ans = (ans + fact[l] * product_poly[l]) % mod

print(ans)