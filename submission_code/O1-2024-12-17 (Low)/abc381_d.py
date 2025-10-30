def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Edge case: if N < 2, the answer can only be 0
    # because we either have no subarray of even length
    # or any single-element subarray doesn't satisfy |X| even.
    if N < 2:
        print(0)
        return

    # B[i] will be the value A[i] if A[i] == A[i+1], else None
    # We will treat consecutive valid (non-None) entries in B as "pairs"
    # The problem reduces to finding a longest contiguous segment in B
    # such that all values are distinct (and none are None).
    B = [None]*(N-1)
    for i in range(N-1):
        if A[i] == A[i+1]:
            B[i] = A[i]

    # Now find the longest subarray of B with no None-values and
    # where all values are distinct.
    # We'll use a two-pointer "sliding window" approach.
    used = {}
    left = 0
    max_pairs = 0
    for right in range(N-1):
        # If B[right] is None, we must reset the window after right
        if B[right] is None:
            # reset
            used.clear()
            left = right + 1
            continue

        val = B[right]
        # If val has appeared in the current window, move left pointer
        # so that the previous occurrence is excluded
        if val in used and used[val] >= left:
            left = used[val] + 1
        used[val] = right

        # Number of pairs in the current window is (right - left + 1)
        curr_pairs = right - left + 1
        if curr_pairs > max_pairs:
            max_pairs = curr_pairs

    # Each pair contributes 2 to the subarray length
    print(max_pairs * 2)

# Do not forget to call main()
if __name__ == "__main__":
    main()