def main():
    import sys, random
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    
    # We'll use a randomized hash approach:
    # For every possible value (they lie in 1..n) assign a random 64-bit integer.
    # Then, the "hash" of a multiset will be the sum (mod 2**64) of the random numbers for the values.
    # Two segments can be rearranged to match (i.e. have the same multiset) if and only if these sums coincide.
    MASK = (1 << 64) - 1
    randVals = [0] * (n + 1)
    for i in range(1, n + 1):
        randVals[i] = random.getrandbits(64)
    
    # Precompute prefix hash arrays for A and B.
    # We use 1-indexing so that for a segment [l, r] the hash is: prefix[r] - prefix[l-1] (mod 2**64).
    prefixA = [0] * (n + 1)
    prefixB = [0] * (n + 1)
    
    for i in range(1, n + 1):
        prefixA[i] = (prefixA[i - 1] + randVals[A[i - 1]]) & MASK
        prefixB[i] = (prefixB[i - 1] + randVals[B[i - 1]]) & MASK
    
    # Process each query.
    out_lines = []
    for _ in range(q):
        l = int(next(it))
        r = int(next(it))
        L = int(next(it))
        R = int(next(it))
        # First, if segments don't have equal length, they can't be rearranged to match.
        if (r - l) != (R - L):
            out_lines.append("No")
            continue
        # Compute hash for the segments. Note: our prefix arrays are 1-indexed.
        hashA = (prefixA[r] - prefixA[l - 1]) & MASK
        hashB = (prefixB[R] - prefixB[L - 1]) & MASK
        out_lines.append("Yes" if hashA == hashB else "No")
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()