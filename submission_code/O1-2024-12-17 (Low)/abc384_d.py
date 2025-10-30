def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, S = int(input_data[0]), int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Sum of one full period
    total = sum(A)
    
    # Quick check: if exactly one full period equals S
    if total == S:
        print("Yes")
        return

    # We will use a classic "sliding‐window/two‐pointer" approach on
    # the array A concatenated with itself (length 2N).  Any subarray
    # of length ≤ 2N in this doubled array can represent either:
    #   (i) A subarray entirely within one period, or
    #  (ii) A subarray crossing exactly one period boundary.
    #
    # If the target sum S can be formed by such a subarray plus some
    # multiple of "total", that means there is a contiguous subsequence
    # in the infinite periodic repetition summing to S.
    #
    # Concretely:
    #   1. Build array B = A + A of length 2N.
    #   2. Use two pointers to find all sub-subarray sums w (non-empty)
    #      of length ≤ 2N that do not exceed S (since we only need sums up
    #      to S when checking S - w).
    #   3. For each such sum w, check if S >= w and (S - w) is a multiple
    #      of total.  If so, we can form S = w + k*total for some k ≥ 0,
    #      implying a valid contiguous subsequence in the infinite sequence.

    B = A + A  # length = 2N

    # Gather all possible sub-subarray sums w <= S (of length up to 2N).
    # We do this with a sliding window.
    sums_set = set()
    left = 0
    curr_sum = 0
    max_len = 2*N
    
    for right in range(2*N):
        curr_sum += B[right]
        # Shrink from the left if sum exceeds S or length exceeds 2N
        while curr_sum > S or (right - left + 1) > max_len:
            curr_sum -= B[left]
            left += 1
        # Now we have a subarray left..right of length ≥ 1 and ≤ 2N
        # whose sum is curr_sum ≤ S
        sums_set.add(curr_sum)

    # Now check if there is w in sums_set such that
    # S >= w and (S - w) % total == 0.
    # If so, answer "Yes" else "No".
    for w in sums_set:
        if w <= S:
            if (S - w) % total == 0:
                print("Yes")
                return

    print("No")


# Call main() at the end
if __name__ == "__main__":
    main()