import sys
data = sys.stdin.read().split()
mod = 998244353
N = int(data[0])
A = [int(x) for x in data[1:N+1]]
M_list = []
i = 0
while i < N:
    val = A[i]
    j = i
    while j < N and A[j] == val:
        j += 1
    L = j - i
    if L % 2 == 1 and L >= 3:
        M_b = (L - 1) // 2
        M_list.append(M_b)
    i = j
K = sum(M_list)
MAX_FACT = 200010
fact = [1] * MAX_FACT
for i in range(1, MAX_FACT):
    fact[i] = (fact[i - 1] * i) % mod
product_W = 1
for M in M_list:
    W_b = 1
    for k in range(1, M + 1):
        W_b = (W_b * (2 * k - 1) % mod) % mod
    product_W = (product_W * W_b % mod) % mod
prod_K_fact = 1
for M in M_list:
    fm = fact[M]
    prod_K_fact = (prod_K_fact * fm % mod) % mod
inv_prod_K_fact = pow(prod_K_fact, mod - 2, mod)
multinomial = (fact[K] * inv_prod_K_fact % mod) % mod
ans = (product_W * multinomial % mod) % mod
print(ans)