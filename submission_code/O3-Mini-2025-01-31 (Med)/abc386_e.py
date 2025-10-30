def main():
    import sys, itertools
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    # Quick cases:
    if K == 1:
        sys.stdout.write(str(max(A)))
        return

    if K == N:
        total = 0
        for x in A:
            total ^= x
        sys.stdout.write(str(total))
        return

    # We use that the number of combinations (C(N, K)) is at most 10^6.
    # It may be beneficial to work with the smaller side.
    #
    # Notice that if K > N/2, then consider choosing the omitted elements.
    # Let total_xor = A_1 XOR A_2 XOR ... XOR A_N.
    # If we omit a subset U of L = N-K elements, then the XOR of the chosen subset is:
    #      total_xor XOR (XOR over U)
    # So, if K > N/2, we enumerate over omitted subsets of size L = N-K.
    
    total_xor = 0
    for x in A:
        total_xor ^= x

    # Decide which side to enumerate.
    use_omitted = False
    if K > N // 2:
        L = N - K
        use_omitted = True
    else:
        L = K

    best = 0
    # Enumerate over combinations of indices of the "smaller" subset
    for comb in itertools.combinations(range(N), L):
        xor_val = 0
        for i in comb:
            xor_val ^= A[i]
        # If we enumerated omitted elements, computing chosen subset XOR:
        candidate = total_xor ^ xor_val if use_omitted else xor_val
        if candidate > best:
            best = candidate

    sys.stdout.write(str(best))


if __name__ == '__main__':
    main()