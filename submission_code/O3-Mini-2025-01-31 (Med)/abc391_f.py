def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    
    # Read the arrays
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    C = [int(next(it)) for _ in range(n)]
    
    # Sort each array in descending order.
    # Because the expression is monotonic in each argument, we know that
    # replacing any index with a higher (i.e. lower-index) element yields a larger value.
    # Hence, the top (k-th largest) value is achieved by indices that are not too far
    # from the beginning.
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    
    # To explain: The expression is
    #   f(i,j,k) = A_i * B_j + B_j * C_k + C_k * A_i
    # which we can rewrite as
    #   f(i,j,k) = A_i * (B_j + C_k) + B_j * C_k.
    # Since A, B, and C are all positive and sorted descending,
    # if you were to replace any index i, j or k with a smaller number (i.e. one from the "top" of the sorted order)
    # the value would only increase.
    #
    # Thus, once we have a guarantee that the number of candidate triples is at least k,
    # it suffices to consider only the top L elements from A, B, and C. In the worst case k can be up to 500,000.
    # We pick L to be the smallest integer (with a slight safety margin) so that L^3 >= k.
    
    L = int(round(k ** (1/3)))
    if L ** 3 < k:
        L += 1
    L += 2   # Safety margin to catch boundary cases
    L = min(n, L)
    
    # Now, because L is very small relative to n (e.g. for k=500000, L will be about 80-82),
    # the k-th largest triple among all n^3 choices is guaranteed to be in the set:
    #   { (i, j, k) : 0 <= i,j,k < L }
    # due to the monotonicity.
    
    # Compute the value f = A_i*B_j + B_j*C_k + C_k*A_i for all i,j,k in [0, L-1]
    # and collect them in a list.
    candidates = []
    for i in range(L):
        a = A[i]
        for j in range(L):
            b = B[j]
            ab = a * b  # precompute a * b
            for k_idx in range(L):
                c = C[k_idx]
                # Compute the triple value.
                # f = A_i*B_j + B_j*C_k + C_k*A_i.
                val = ab + b * c + c * a
                candidates.append(val)
    
    # Sort the candidate triple values in descending order.
    candidates.sort(reverse=True)
    # The k-th largest value is at index k-1 (since Python lists are 0-indexed).
    ans = candidates[k - 1]
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()