def main():
    import sys
    from itertools import combinations

    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # If picking exactly 1 element, the best XOR is just the max element.
    if K == 1:
        print(max(A))
        return

    # If picking all elements, the XOR is simply the XOR of the entire array.
    if K == N:
        total_xor = 0
        for x in A:
            total_xor ^= x
        print(total_xor)
        return

    # Precompute the XOR of all elements.
    total_xor = 0
    for x in A:
        total_xor ^= x

    # Since C(N,K) = C(N, N-K), we choose the smaller subset size to iterate over.
    # If we choose r = min(K, N-K), then each subset of size r corresponds
    # to its complement of size N-r which is either the chosen set or not.
    r = min(K, N-K)
    max_xor = 0

    # Enumerate all subsets of size r.  If K == r, we'll XOR those subsets directly;
    # if K != r, we'll XOR the complement by XORing total_xor with the subset's XOR.
    if r == K:
        # We pick subsets of size K directly
        for combo in combinations(A, r):
            xor_val = 0
            for val in combo:
                xor_val ^= val
            if xor_val > max_xor:
                max_xor = xor_val
    else:
        # We pick subsets of size N-K; the chosen K subset is the complement
        for combo in combinations(A, r):
            xor_val = 0
            for val in combo:
                xor_val ^= val
            # XOR of the chosen K elements is total_xor ^ xor_val
            complement_xor = total_xor ^ xor_val
            if complement_xor > max_xor:
                max_xor = complement_xor

    print(max_xor)

# Do not forget to call main()
main()