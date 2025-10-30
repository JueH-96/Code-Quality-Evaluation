def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # We will use a single-pass approach to count the number of arithmetic subarrays.
    # curr_len keeps track of the length of the current arithmetic subarray ending at index i.
    # We add curr_len to the answer for each i.

    if N == 1:
        print(1)
        return

    answer = 0
    curr_len = 1  # At least one element is always an arithmetic subarray.

    answer += curr_len  # Account for the subarray [A[0]].

    for i in range(1, N):
        if i == 1:
            # The subarray of length 2 is always arithmetic by definition,
            # as we only have one difference to consider.
            curr_len = 2
        else:
            # If the difference matches, extend the arithmetic subarray, otherwise reset.
            if (A[i] - A[i - 1]) == (A[i - 1] - A[i - 2]):
                curr_len += 1
            else:
                curr_len = 2
        answer += curr_len

    print(answer)

# Do not remove the following call
main()