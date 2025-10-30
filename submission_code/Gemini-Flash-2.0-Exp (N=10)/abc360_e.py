def mod_inv(a, m):
    return pow(a, m - 2, m)

def solve():
    n, k = map(int, input().split())
    mod = 998244353
    
    expected_pos = 1
    
    for _ in range(k):
        expected_pos = (expected_pos * (n - 2) * mod_inv(n, mod) + (n + 1) * mod_inv(n, mod)) % mod
    
    print(expected_pos)

solve()