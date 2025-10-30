def main():
    import sys, random
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]

    # We assign two random values (mod different primes) for each number in [1, N].
    # Their sums over a segment will serve as a hash of the multiset of that segment.
    # Two segments can be rearranged into each other exactly when they have the same frequency counts,
    # which is equivalent to having identical hash values (with high probability).
    mod1 = 10**9 + 7
    mod2 = 10**9 + 9
    random.seed(0)
    hash_for_val_mod1 = [0] * (N + 1)
    hash_for_val_mod2 = [0] * (N + 1)
    for x in range(1, N+1):
        hash_for_val_mod1[x] = random.randrange(1, mod1)
        hash_for_val_mod2[x] = random.randrange(1, mod2)

    # Build prefix sums for A and B that will help us compute the hash over any subinterval fast.
    prefixA1 = [0] * (N + 1)
    prefixA2 = [0] * (N + 1)
    prefixB1 = [0] * (N + 1)
    prefixB2 = [0] * (N + 1)
    for i in range(1, N + 1):
        prefixA1[i] = (prefixA1[i-1] + hash_for_val_mod1[A[i-1]]) % mod1
        prefixA2[i] = (prefixA2[i-1] + hash_for_val_mod2[A[i-1]]) % mod2
        prefixB1[i] = (prefixB1[i-1] + hash_for_val_mod1[B[i-1]]) % mod1
        prefixB2[i] = (prefixB2[i-1] + hash_for_val_mod2[B[i-1]]) % mod2

    # Process each query. Two segments can be rearranged to match if and only if:
    # 1. They have the same length.
    # 2. Their multisets (frequency counts) are identical.
    # We check condition (2) by comparing our two independent hash sums.
    answers = []
    for _ in range(Q):
        l = int(next(it))
        r = int(next(it))
        L = int(next(it))
        R = int(next(it))
        # Check length: note that segment lengths are (r-l+1) and (R-L+1)
        if (r - l) != (R - L):
            answers.append("No")
        else:
            # Compute the hash for A[l..r]
            hashA1 = (prefixA1[r] - prefixA1[l-1]) % mod1
            hashA2 = (prefixA2[r] - prefixA2[l-1]) % mod2
            # Compute the hash for B[L..R]
            hashB1 = (prefixB1[R] - prefixB1[L-1]) % mod1
            hashB2 = (prefixB2[R] - prefixB2[L-1]) % mod2
            if hashA1 == hashB1 and hashA2 == hashB2:
                answers.append("Yes")
            else:
                answers.append("No")
    sys.stdout.write("
".join(answers))


if __name__ == "__main__":
    main()