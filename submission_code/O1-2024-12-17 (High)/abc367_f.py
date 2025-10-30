def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Fast integer iterator over the input tokens:
    idx = 0
    def read_int():
        nonlocal idx
        val = int(input_data[idx])
        idx += 1
        return val
    
    # Read N and Q
    N = read_int()
    Q = read_int()
    
    # Read sequences A and B (0-based indexing in Python)
    A = [read_int() for _ in range(N)]
    B = [read_int() for _ in range(N)]
    
    # Two large prime moduli for double hashing
    M1, M2 = 10**9+7, 10**9+9
    
    # ------------------------------------------------------------------------
    # We'll create two pseudo-random generators for assigning hash factors
    # to each possible value in [1..N].  This avoids Python's slower random.
    # You can tweak seeds to any fixed values you like.
    # ------------------------------------------------------------------------
    seed1 = 0x5A5A5A5A
    def rand1():
        nonlocal seed1
        # Simple 32-bit linear congruential generator
        seed1 = (seed1 * 0xdefaced + 1) & 0xffffffff
        return seed1
    
    seed2 = 0xDEADBEEF
    def rand2():
        nonlocal seed2
        seed2 = (seed2 * 0xf00dbeef + 3) & 0xffffffff
        return seed2
    
    # ------------------------------------------------------------------------
    # Build arrays val1, val2 of size N+1, where val1[x] and val2[x] are the
    # hash multipliers for the integer x.  Integers range in [1..N].
    # We ensure they are in [1..(M-1)] so we never multiply by zero mod M.
    # ------------------------------------------------------------------------
    val1 = [0]*(N+1)
    val2 = [0]*(N+1)
    for i in range(1, N+1):
        val1[i] = (rand1() % (M1-1)) + 1
        val2[i] = (rand2() % (M2-1)) + 1
    
    # ------------------------------------------------------------------------
    # Build prefix products for A and B, for each modulus.
    # prefixA1[i] = (val1[A[0]] * val1[A[1]] * ... * val1[A[i-1]]) mod M1
    # prefixB1[i] similarly for B, and likewise for M2.
    #
    # Then we'll build a backward prefix to store inverses so we can do
    # subarray-hash = prefix[r] * inv(prefix[l]) in O(1).
    # ------------------------------------------------------------------------
    prefixA1 = [0]*(N+1)
    prefixA2 = [0]*(N+1)
    prefixB1 = [0]*(N+1)
    prefixB2 = [0]*(N+1)
    
    # Forward pass (build prefix products)
    prefixA1[0] = 1
    prefixA2[0] = 1
    prefixB1[0] = 1
    prefixB2[0] = 1
    
    for i in range(N):
        prefixA1[i+1] = (prefixA1[i] * val1[A[i]]) % M1
        prefixA2[i+1] = (prefixA2[i] * val2[A[i]]) % M2
        prefixB1[i+1] = (prefixB1[i] * val1[B[i]]) % M1
        prefixB2[i+1] = (prefixB2[i] * val2[B[i]]) % M2
    
    # We'll store inverses in these arrays
    prefixAinv1 = [0]*(N+1)
    prefixAinv2 = [0]*(N+1)
    prefixBinv1 = [0]*(N+1)
    prefixBinv2 = [0]*(N+1)
    
    # Compute inverse of prefixA1[N] and prefixA2[N] by Fermat's little theorem (since M is prime).
    # inv(x) = x^(M-2) mod M,  done in Python by pow(x, M-2, M).
    prefixAinv1[N] = pow(prefixA1[N], M1-2, M1)
    prefixAinv2[N] = pow(prefixA2[N], M2-2, M2)
    prefixBinv1[N] = pow(prefixB1[N], M1-2, M1)
    prefixBinv2[N] = pow(prefixB2[N], M2-2, M2)
    
    # Backward pass to fill inverses
    for i in range(N, 0, -1):
        # A's inverses
        xA = A[i-1]  # the integer in A at position i-1
        prefixAinv1[i-1] = (prefixAinv1[i] * val1[xA]) % M1
        prefixAinv2[i-1] = (prefixAinv2[i] * val2[xA]) % M2
        
        # B's inverses
        xB = B[i-1]
        prefixBinv1[i-1] = (prefixBinv1[i] * val1[xB]) % M1
        prefixBinv2[i-1] = (prefixBinv2[i] * val2[xB]) % M2
    
    # ------------------------------------------------------------------------
    # Now we can answer queries. For a subarray [l..r] in 1-based indexing,
    # the hash in M1 is: hashA1 = prefixA1[r] * prefixAinv1[l-1] mod M1
    # similarly for M2, and likewise for B with L and R.
    #
    # We first check if (r-l) == (R-L). If not equal => "No".
    # If equal, compare (hashA1, hashA2) vs (hashB1, hashB2).  If equal => "Yes", else "No".
    # ------------------------------------------------------------------------
    
    out = []
    for _ in range(Q):
        l = read_int()
        r = read_int()
        L = read_int()
        R = read_int()
        
        lengthA = r - l + 1
        lengthB = R - L + 1
        if lengthA != lengthB:
            out.append("No")
            continue
        
        # 1-based subarray [l..r] =>  prefixA1[r], prefixAinv1[l-1]
        hashA1 = (prefixA1[r] * prefixAinv1[l-1]) % M1
        hashA2 = (prefixA2[r] * prefixAinv2[l-1]) % M2
        
        hashB1 = (prefixB1[R] * prefixBinv1[L-1]) % M1
        hashB2 = (prefixB2[R] * prefixBinv2[L-1]) % M2
        
        if hashA1 == hashB1 and hashA2 == hashB2:
            out.append("Yes")
        else:
            out.append("No")
    
    # Print all answers
    print("
".join(out))

# Do not forget to call main!
main()