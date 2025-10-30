def modinv(a, p):
    return pow(a, p - 2, p)

def comb(n, k, mod):
    if n < k or k < 0:
        return 0
    return (fact[n] * inv_fact[k] % mod) * inv_fact[n - k] % mod

def precompute_factorials(max_n, mod):
    fact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = modinv(fact[max_n], mod)
    for i in range(max_n - 1, 0, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod
    return fact, inv_fact

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    W, H, L, R, D, U = map(int, data)
    
    MOD = 998244353
    
    # Precompute factorials and inverse factorials up to W+H
    global fact, inv_fact
    fact, inv_fact = precompute_factorials(W + H, MOD)
    
    # Calculate the number of paths
    total_paths = 0
    
    # Paths in the unrestricted regions
    # 1. x < L
    for x in range(L):
        total_paths += comb(x + H, x, MOD)
        total_paths %= MOD
    
    # 2. R < x
    for x in range(R + 1, W + 1):
        total_paths += comb(x + H, x, MOD)
        total_paths %= MOD
    
    # 3. y < D
    for y in range(D):
        total_paths += comb(W + y, y, MOD)
        total_paths %= MOD
    
    # 4. U < y
    for y in range(U + 1, H + 1):
        total_paths += comb(W + y, y, MOD)
        total_paths %= MOD
    
    # Subtract the overcounted paths in the corners
    # 1. x < L and y < D
    if L > 0 and D > 0:
        for x in range(L):
            for y in range(D):
                total_paths -= comb(x + y, x, MOD)
                total_paths %= MOD
    
    # 2. x < L and U < y
    if L > 0 and U < H:
        for x in range(L):
            for y in range(U + 1, H + 1):
                total_paths -= comb(x + y, x, MOD)
                total_paths %= MOD
    
    # 3. R < x and y < D
    if R < W and D > 0:
        for x in range(R + 1, W + 1):
            for y in range(D):
                total_paths -= comb(x + y, x, MOD)
                total_paths %= MOD
    
    # 4. R < x and U < y
    if R < W and U < H:
        for x in range(R + 1, W + 1):
            for y in range(U + 1, H + 1):
                total_paths -= comb(x + y, x, MOD)
                total_paths %= MOD
    
    print(total_paths)

if __name__ == "__main__":
    main()