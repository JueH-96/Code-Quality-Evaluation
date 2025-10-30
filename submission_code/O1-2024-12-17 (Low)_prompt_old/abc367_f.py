def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO index
    idx = 0
    
    # Read N and Q
    N = int(input_data[idx]); idx+=1
    Q = int(input_data[idx]); idx+=1

    # Read A array (1-based indexing for convenience)
    A = [0]*(N+1)
    for i in range(1, N+1):
        A[i] = int(input_data[idx])
        idx += 1

    # Read B array
    B = [0]*(N+1)
    for i in range(1, N+1):
        B[i] = int(input_data[idx])
        idx += 1

    # Two large primes for double hashing
    MOD1, MOD2 = 10**9+7, 10**9+9
    # You may choose your base(s) in a suitable range
    BASE1, BASE2 = 10007, 20011

    # --------------------------------------------------------------------
    # Precompute powers for BASE1 and BASE2 up to N (since A[i], B[i] ≤ N)
    # pow1[x] = BASE1^x % MOD1
    # pow2[x] = BASE2^x % MOD2
    # Also compute inverses invPow1[x], invPow2[x] using Fermat's little theorem
    # --------------------------------------------------------------------
    pow1 = [1]*(N+1)
    pow2 = [1]*(N+1)
    for i in range(1, N+1):
        pow1[i] = (pow1[i-1] * BASE1) % MOD1
        pow2[i] = (pow2[i-1] * BASE2) % MOD2

    # Function for fast exponentiation mod
    def modexp(base, exp, mod):
        result = 1
        cur = base
        e = exp
        while e > 0:
            if e & 1:
                result = (result * cur) % mod
            cur = (cur * cur) % mod
            e >>= 1
        return result

    # Precompute inverses of each power
    invPow1 = [1]*(N+1)
    invPow2 = [1]*(N+1)
    invPow1[N] = modexp(pow1[N], MOD1-2, MOD1)
    invPow2[N] = modexp(pow2[N], MOD2-2, MOD2)
    # Fill downwards to avoid repeated exponentiation
    # invPow1[k] = (BASE1^k)^{-1} = (BASE1^{k+1})^{-1} * BASE1
    for k in range(N-1, -1, -1):
        invPow1[k] = (invPow1[k+1] * BASE1) % MOD1
        invPow2[k] = (invPow2[k+1] * BASE2) % MOD2

    # --------------------------------------------------------------------
    # Build prefix-hash arrays for A and B in both moduli
    # hashA1[i] = product of BASE1^(A[j]) for j=1..i (mod MOD1)
    # hashA2[i] = product of BASE2^(A[j]) for j=1..i (mod MOD2)
    # similarly for B
    #
    # Also store their inverses so we can get sub-interval products in O(1).
    # hashA1[r]*invHashA1[l-1] -> product of A[l..r].
    # --------------------------------------------------------------------
    hashA1 = [1]*(N+1)
    hashA2 = [1]*(N+1)
    hashB1 = [1]*(N+1)
    hashB2 = [1]*(N+1)

    invHashA1 = [1]*(N+1)
    invHashA2 = [1]*(N+1)
    invHashB1 = [1]*(N+1)
    invHashB2 = [1]*(N+1)

    for i in range(1, N+1):
        hashA1[i] = (hashA1[i-1] * pow1[A[i]]) % MOD1
        hashA2[i] = (hashA2[i-1] * pow2[A[i]]) % MOD2
        hashB1[i] = (hashB1[i-1] * pow1[B[i]]) % MOD1
        hashB2[i] = (hashB2[i-1] * pow2[B[i]]) % MOD2

        invHashA1[i] = (invHashA1[i-1] * invPow1[A[i]]) % MOD1
        invHashA2[i] = (invHashA2[i-1] * invPow2[A[i]]) % MOD2
        invHashB1[i] = (invHashB1[i-1] * invPow1[B[i]]) % MOD1
        invHashB2[i] = (invHashB2[i-1] * invPow2[B[i]]) % MOD2

    # Helper to get product-hash of A[l..r] in each mod
    def get_subhash_A(l, r):
        if l == 1:
            # full from start
            return (hashA1[r], hashA2[r])
        # product in mod1:
        val1 = (hashA1[r] * invHashA1[l-1]) % MOD1
        # product in mod2:
        val2 = (hashA2[r] * invHashA2[l-1]) % MOD2
        return (val1, val2)

    def get_subhash_B(l, r):
        if l == 1:
            return (hashB1[r], hashB2[r])
        val1 = (hashB1[r] * invHashB1[l-1]) % MOD1
        val2 = (hashB2[r] * invHashB2[l-1]) % MOD2
        return (val1, val2)

    out = []
    for _ in range(Q):
        l_i = int(input_data[idx]); idx+=1
        r_i = int(input_data[idx]); idx+=1
        L_i = int(input_data[idx]); idx+=1
        R_i = int(input_data[idx]); idx+=1

        # First check length
        if (r_i - l_i) != (R_i - L_i):
            out.append("No")
            continue

        # Compare double-hash (unordered product) for sub-intervals
        hA = get_subhash_A(l_i, r_i)
        hB = get_subhash_B(L_i, R_i)
        if hA == hB:
            out.append("Yes")
        else:
            out.append("No")

    print("
".join(out))

# Let's call solve() to execute.
# solve()