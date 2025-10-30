def modinv(a, p):
    return pow(a, p - 2, p)

def precompute_factorials_and_inverses(max_n, mod):
    fact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = fact[i - 1] * i % mod
    
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = modinv(fact[max_n], mod)
    for i in range(max_n - 1, 0, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod
    
    return fact, inv_fact

def binomial(n, k, fact, inv_fact, mod):
    if k > n or k < 0:
        return 0
    return fact[n] * inv_fact[k] % mod * inv_fact[n - k] % mod

def count_paths(x, y, fact, inv_fact, mod):
    return binomial(x + y, x, fact, inv_fact, mod)

def main():
    import sys
    input = sys.stdin.read
    W, H, L, R, D, U = map(int, input().strip().split())
    
    mod = 998244353
    max_n = W + H
    
    fact, inv_fact = precompute_factorials_and_inverses(max_n, mod)
    
    total_paths = 0
    
    # Count paths in each region
    # 1. Left of the rectangle
    for y in range(H + 1):
        total_paths += count_paths(L - 1, y, fact, inv_fact, mod)
        total_paths %= mod
    
    # 2. Right of the rectangle
    for y in range(H + 1):
        total_paths += count_paths(W - R, y, fact, inv_fact, mod)
        total_paths %= mod
    
    # 3. Below the rectangle
    for x in range(L, R + 1):
        total_paths += count_paths(x, D - 1, fact, inv_fact, mod)
        total_paths %= mod
    
    # 4. Above the rectangle
    for x in range(L, R + 1):
        total_paths += count_paths(x, H - U, fact, inv_fact, mod)
        total_paths %= mod
    
    print(total_paths)

if __name__ == "__main__":
    main()