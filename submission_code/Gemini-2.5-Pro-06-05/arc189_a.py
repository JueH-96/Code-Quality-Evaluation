import sys

# Set a higher recursion limit for potentially deep recursion in NTT, though not strictly needed for iterative versions.
sys.setrecursionlimit(2 * 10**5 + 5)

# Fast I/O
input = sys.stdin.readline

# --- MODULAR ARITHMETIC ---
MOD = 998244353
G = 3 # Primitive root for MOD

def power(a, b):
    """Computes a^b % MOD."""
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

def inv(n):
    """Computes modular inverse of n."""
    return power(n, MOD - 2)

# --- NTT IMPLEMENTATION ---
def ntt(a, invert):
    """Performs Number Theoretic Transform."""
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = power(G, (MOD - 1) // length)
        if invert:
            wlen = inv(wlen)
        i = 0
        while i < n:
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = (a[i + j + length // 2] * w) % MOD
                a[i + j] = (u + v) % MOD
                a[i + j + length // 2] = (u - v + MOD) % MOD
                w = (w * wlen) % MOD
            i += length
        length <<= 1
    
    if invert:
        n_inv = inv(n)
        for i in range(n):
            a[i] = (a[i] * n_inv) % MOD

def multiply(a, b):
    """Multiplies two polynomials using NTT."""
    n = 1
    while n < len(a) + len(b):
        n <<= 1
    
    fa = a + [0] * (n - len(a))
    fb = b + [0] * (n - len(b))
    
    ntt(fa, False)
    ntt(fb, False)
    
    for i in range(n):
        fa[i] = (fa[i] * fb[i]) % MOD
        
    ntt(fa, True)
    return fa

# --- POLYNOMIAL OPERATIONS ---
def poly_inv(a, deg):
    """Computes inverse of polynomial a up to degree deg."""
    if deg == 0:
        return [inv(a[0])]
    
    b = poly_inv(a, (deg - 1) // 2)
    n = 1
    while n < 2 * deg + 1:
        n <<= 1
    
    a_resized = a[:deg + 1] + [0] * (n - (deg + 1))
    b_resized = b + [0] * (n - len(b))
    
    ntt(a_resized, False)
    ntt(b_resized, False)
    
    for i in range(n):
        b_resized[i] = (b_resized[i] * (2 - a_resized[i] * b_resized[i] % MOD + MOD)) % MOD
        
    ntt(b_resized, True)
    return b_resized[:deg + 1]

def poly_sqrt(a, deg):
    """Computes square root of polynomial a up to degree deg."""
    if deg == 0:
        return [1]

    b = poly_sqrt(a, (deg - 1) // 2)
    b_inv = poly_inv(b, deg)
    
    a_resized = a[:deg + 1]
    
    res = multiply(a_resized, b_inv)
    res = res[:deg+1]
    
    for i in range(len(b)):
        res[i] = (res[i] + b[i]) % MOD
    
    inv2 = inv(2)
    for i in range(deg + 1):
        res[i] = (res[i] * inv2) % MOD
        
    return res

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    if A[0] != 1 or A[-1] != (N % 2):
        print(0)
        return

    blocks = []
    if N > 0:
        start = 0
        for i in range(1, N):
            if A[i] != A[i-1]:
                blocks.append(i - start)
                start = i
        blocks.append(N - start)

    total_ops = 0
    num_ops_list = []
    
    current_pos = 1
    current_val = 1
    for k in blocks:
        if k % 2 == 0:
            print(0)
            return
        
        # B_i = 1 if i is odd, 0 if i is even.
        if (current_pos % 2) != current_val:
            print(0)
            return

        ops_for_block = (k - 1) // 2
        num_ops_list.append(ops_for_block)
        total_ops += ops_for_block
        current_pos += k
        current_val = 1 - current_val

    max_ops = (N - 1) // 2
    if max_ops == 0:
        print(1)
        return
    
    deg = max_ops + 1
    
    ln_one_minus_x = [0] * deg
    for i in range(1, deg):
        ln_one_minus_x[i] = (MOD - inv(i)) % MOD

    P = [0] * deg
    P[0] = 1
    for i in range(1, deg):
        P[i] = (4 * ln_one_minus_x[i]) % MOD
    
    Q = poly_sqrt(P, deg - 1)

    Y = [0] * deg
    Y[0] = (1 - Q[0] + MOD) % MOD
    for i in range(1, deg):
        Y[i] = (MOD - Q[i]) % MOD

    inv2 = inv(2)
    for i in range(deg):
        Y[i] = (Y[i] * inv2) % MOD
    
    h = Y

    fact = [1] * (N + 1)
    for i in range(2, N + 1):
        fact[i] = (fact[i-1] * i) % MOD

    ans = fact[total_ops]
    for ops in num_ops_list:
        if ops >= len(h):
            print(0)
            return
        ans = (ans * h[ops]) % MOD
    
    print(ans)

solve()