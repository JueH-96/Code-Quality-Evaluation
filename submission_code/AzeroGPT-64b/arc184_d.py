import math

MOD = 998244353

def pow_with_mod(m, n, mod):
    result = 1
    while n > 0:
        if n & 1:
            result = (result * m) % mod
        m = (m * m) % mod
        n >>= 1
    return result

def inv_with_mod(n, mod):
    return pow_with_mod(n, mod-2, mod)

def nCk_mod(n, k, mod):
    k = min(k, n-k)
    numer = denom = 1
    for i in range(k):
        numer = (numer * (n-i)) % mod
        denom = (denom * (i+1)) % mod
    return (numer * inv_with_mod(denom, mod)) % mod

N = int(input())
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

left = []
right = []

for left_index in range(N):
    xL, yL = points[left_index]
    count = 0
    for right_index in range(left_index+1, N):
        xR, yR = points[right_index]
        if (xL < xR and yL < yR) or (xL > xR and yL > yR):
            count += 1
    if left_index == 0 or points[left_index-1][0] < points[left_index][0]:
        left.append(count)
    else:
        right.append(count)

left_sum = sum(left)
right_sum = sum(right)

def map_sum(input_list, map_list):
    D = {}
    for value, map_value in zip(input_list, map_list):
        if not map_value in D:
            D[map_value] = 0
        D[map_value] += value
    return D

D = map_sum(right, left)

ans = pow_with_mod(2, left_sum, MOD)

for k, v in D.items():
    ans = (ans + nCk_mod(N-1, k, MOD) * pow_with_mod(2, v, MOD) * inv_with_mod(2, left_sum+1, MOD)) % MOD
print(ans)