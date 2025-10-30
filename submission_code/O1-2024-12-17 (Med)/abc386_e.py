def main():
    import sys
    import itertools

    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # If we XOR all elements:
    xor_all = 0
    for val in A:
        xor_all ^= val

    # We will choose min(K, N-K) elements to iterate over (since C(N,K) = C(N, N-K)),
    # then adjust the final XOR accordingly if we were actually choosing the other size.
    chosen_size = min(K, N - K)

    max_xor = 0
    if chosen_size == 0:
        # Edge case: if K=0 or K=N, then there's only one way to choose.
        # If K=0, XOR is 0. If K=N, XOR is xor_all.
        max_xor = xor_all if K == N else 0
    else:
        # Enumerate all subsets of size 'chosen_size' and compute their XOR.
        # If chosen_size == K, we track that XOR directly.
        # Otherwise (chosen_size == N-K), the XOR of the K-chosen set is xor_all ^ subset_xor.
        if chosen_size == K:
            # Directly compute XOR of chosen_size elements from A
            for combo in itertools.combinations(A, chosen_size):
                x = 0
                for c in combo:
                    x ^= c
                if x > max_xor:
                    max_xor = x
        else:
            # We are enumerating subsets of size N-K, so the chosen K's XOR = xor_all ^ subset_xor
            for combo in itertools.combinations(A, chosen_size):
                x = 0
                for c in combo:
                    x ^= c
                candidate = xor_all ^ x
                if candidate > max_xor:
                    max_xor = candidate

    print(max_xor)

if __name__ == "__main__":
    main()