def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast parsing
    N, Q = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+2*N]))
    query_data = input_data[2+2*N:]
    
    # We will use double hashing (two moduli) to reduce collision probability.
    # The idea: For each array, we build a prefix array of the form:
    # prefixA[i] = sum_{k=1..i} p^(A[k]) mod M
    # Then the hash of a subarray (l..r) is prefixA[r] - prefixA[l-1] (mod M),
    # which equals the sum of p^(A[x]) for x in [l..r].
    #
    # If subarray A[l..r] is a rearrangement of subarray B[L..R], they must have
    # identical multisets of elements, so their sums of p^(value) should match
    # (mod M). We do this under two distinct large moduli to drastically lower
    # collision chances.
    
    # Choose two pairs of (base, mod). We'll precompute p^i for i up to N (the max value in A,B≤N).
    # Because A[i], B[i] ≤ N, we only need p^(k) for k from 1..N, not for indices but for the values themselves.
    
    # Large prime moduli:
    M1, M2 = 10**9+7, 10**9+9
    # Random-ish bases (must be large enough to reduce collisions, but also feasible):
    p1, p2 = 1000003, 1000033
    
    # Precompute powers for value exponents up to N
    pow1_mod1 = [0]*(N+1)
    pow1_mod2 = [0]*(N+1)
    pow1_mod1[0] = pow1_mod2[0] = 1
    for i in range(1, N+1):
        pow1_mod1[i] = (pow1_mod1[i-1] * p1) % M1
        pow1_mod2[i] = (pow1_mod2[i-1] * p2) % M2
    
    # Build prefix-hash arrays for A and B
    # AHash1[i] = sum_{k=1..i} p^(A[k-1]) mod M1  (k in code is 1-based, A[k-1] is the actual value)
    # likewise AHash2, BHash1, BHash2
    AHash1 = [0]*(N+1)
    AHash2 = [0]*(N+1)
    BHash1 = [0]*(N+1)
    BHash2 = [0]*(N+1)
    
    for i in range(1, N+1):
        AHash1[i] = (AHash1[i-1] + pow1_mod1[A[i-1]]) % M1
        AHash2[i] = (AHash2[i-1] + pow1_mod2[A[i-1]]) % M2
        BHash1[i] = (BHash1[i-1] + pow1_mod1[B[i-1]]) % M1
        BHash2[i] = (BHash2[i-1] + pow1_mod2[B[i-1]]) % M2
    
    # Now process each query
    idx = 0
    out = []
    for _ in range(Q):
        l, r, L, R = map(int, query_data[idx:idx+4])
        idx += 4
        
        # Quick check: if lengths differ, answer No
        if (r - l) != (R - L):
            out.append("No")
            continue
        
        # Subarray hash for A[l..r]
        subA1 = (AHash1[r] - AHash1[l-1]) % M1
        subA2 = (AHash2[r] - AHash2[l-1]) % M2
        # Subarray hash for B[L..R]
        subB1 = (BHash1[R] - BHash1[L-1]) % M1
        subB2 = (BHash2[R] - BHash2[L-1]) % M2
        
        if (subA1 == subB1) and (subA2 == subB2):
            out.append("Yes")
        else:
            out.append("No")
    
    print("
".join(out))

# Don't forget to call main!
if __name__ == "__main__":
    main()