# YOUR CODE HERE
MOD = 998244353

def pow_mod(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def inv_mod(x, p):
    return pow_mod(x, p - 2, p)

n, k = map(int, input().split())

inv_n = inv_mod(n, MOD)
inv_n_sq = (inv_n * inv_n) % MOD
prob_not_moved = (n - 1) * (n - 1) * inv_n_sq % MOD
prob_moved = 1 - prob_not_moved + MOD
prob_moved %= MOD

expected_pos = 1
for _ in range(k):
    expected_pos = (expected_pos * prob_not_moved + (expected_pos + prob_moved) * inv_n * (n - 1) ) % MOD

print(expected_pos)