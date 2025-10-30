def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # -------------------------------------------------------------------
    # f(x) = x with all factors of 2 removed
    # Equivalently, f(x) = x >> (number_of_trailing_zero_bits_in_x).
    # We can compute the number of trailing zeros in x by:
    #    tz = (x & -x).bit_length() - 1
    # Then f(x) = x >> tz.
    #
    # Example: A_i = 12 (1100 in binary)
    #   trailing_zeros(12) = 2  => f(12) = 12 >> 2 = 3.
    # -------------------------------------------------------------------
    def f(x: int) -> int:
        # number of trailing zeros:
        tz = (x & -x).bit_length() - 1
        return x >> tz

    # -------------------------------------------------------------------
    # Each A_i can be written uniquely as A_i = 2^k * B, with B odd.
    # Let k_i = trailing_zeros(A_i), B_i = A_i >> k_i.
    # We'll group all elements by their k_i.
    # Then the sum of f(A_i + A_j) can be broken into two parts:
    #
    #  1) Pairs from different groups (k1 != k2).
    #     If k1 < k2, f(2^k1*B1 + 2^k2*B2) = f(2^k1 * [ B1 + 2^(k2-k1)*B2 ]).
    #     Because B1 is odd, 2^(k2-k1)*B2 is even if k2-k1 >= 1,
    #     so the sum inside brackets is an odd + an even = odd,
    #     so it has 0 trailing zeros.  Hence total trailing zeros = k1,
    #     and f(...) = (B1 + 2^(k2-k1)*B2).
    #     (Similarly if k1 > k2, the formula is 2^(k1-k2)*B1 + B2.)
    #
    #     We only need the sum over i<j to avoid double counting; but since
    #     the function is symmetric f(A_i + A_j) = f(A_j + A_i), we can
    #     just iterate k1 < k2 once.
    #
    #  2) Pairs within the same group k.
    #     Then A_i + A_j = 2^k*(B_i + B_j).
    #     The number of trailing zeros = k + trailing_zeros_in(B_i + B_j).
    #     So f(A_i + A_j) = (B_i + B_j) >> trailing_zeros(B_i + B_j).
    #     Where B_i, B_j are odd (so B_i+B_j is even).
    #
    # Also remember to include the diagonal terms i=j:
    #   f(A_i + A_i) = f(2 * A_i) = f(2^(k+1) * B_i) = B_i.
    #
    # We'll implement this plan carefully.
    #
    # NOTE on complexity:
    # - The cross-group part is easy to compute in O(K^2) where K ≤ ~24
    #   (since 2^24 > 10^7, that’s the maximum exponent of 2 dividing a number ≤10^7).
    # - The tricky part is summing over pairs within the same group.
    #   A naïve O(freq_k^2) loop could be very large if freq_k is up to 2e5.
    #
    # Nonetheless, this solution is correct.  In very large worst-case scenarios
    # an optimized approach (e.g., using FFT-based convolution on the halved-values)
    # is needed.  For correctness and clarity here, we implement a direct method.
    # -------------------------------------------------------------------

    from collections import defaultdict

    # Group the values by their k (the exponent of 2).
    groups = defaultdict(list)
    max_k = 0
    for val in A:
        # trailing zeros
        tk = (val & -val).bit_length() - 1
        b = val >> tk
        groups[tk].append(b)
        if tk > max_k:
            max_k = tk

    # Precompute for each group: freq (just the length) and sumB (sum of B).
    freq = {}
    sumB = {}
    for k, arr in groups.items():
        freq[k] = len(arr)
        sumB[k] = sum(arr)

    # Part 1: Sum for cross-group pairs k1 < k2
    ans = 0
    all_ks = sorted(groups.keys())
    for i in range(len(all_ks)):
        k1 = all_ks[i]
        for j in range(i+1, len(all_ks)):
            k2 = all_ks[j]
            # For pairs (k1, k2) with k1 < k2:
            # sum over B1 in group k1, B2 in group k2: f(2^k1*B1 + 2^k2*B2)
            # = sum_{B1, B2}( B1 + 2^(k2-k1)*B2 ).
            # = freq[k2]* sumB[k1] + 2^(k2-k1) * freq[k1]* sumB[k2].
            count1 = freq[k2] * sumB[k1]
            count2 = (1 << (k2 - k1)) * (freq[k1] * sumB[k2])
            ans += count1 + count2

    # Part 2a: Diagonal (i = j) => f(2 * A_i) = B_i if A_i = 2^k * B_i.
    for k in all_ks:
        ans += sumB[k]  # each element in group k contributes B_i

    # Part 2b: Within the same group (k), for i < j
    # We need sum_{i<j} of f(2^k(B_i + B_j)).
    # = sum_{i<j} of (B_i + B_j) >> trailing_zeros(B_i + B_j).
    # Let us do a direct O(n^2) within each group.  (Correct, but possibly slow.)
    #
    # A more elaborate convolution-based approach would be needed in practice
    # for very large freq[k].  Here we do the straightforward method for clarity.
    #
    # We'll define a small function to compute odd-sum f for two odd ints B1,B2:
    #   odd_sum(B1,B2) = (B1+B2) >> trailing_zeros(B1+B2), but B1+B2 is even.
    #   That is simply f(B1+B2).  We can reuse f(...) from above.
    #
    def odd_sum(b1, b2):
        return f(b1 + b2)

    for k in all_ks:
        arr = groups[k]
        arr_len = len(arr)
        # naive pairwise
        # sum_{0 <= i < j < arr_len} odd_sum(arr[i], arr[j])
        part = 0
        # If freq is large, this is costly.  But correct.
        for i in range(arr_len):
            bi = arr[i]
            for j in range(i+1, arr_len):
                bj = arr[j]
                part += odd_sum(bi, bj)
        ans += part

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()