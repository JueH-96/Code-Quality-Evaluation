def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:1+N]))
    B = list(map(int, input_data[1+N:1+N+(N-1)]))

    # Sort toy sizes and box sizes in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)

    i = 0  # index over toys
    j = 0  # index over boxes
    skip_used = False
    unmatched_toy_index = -1  # which toy is "skipped" (goes to new box)

    # Match as many toys as possible using largest-to-largest strategy.
    # We have N toys and N-1 boxes, so we can afford to skip exactly one toy.
    while i < N and j < (N - 1):
        if B[j] >= A[i]:
            # We can match this toy with the current box
            j += 1
            i += 1
        else:
            # Cannot match A[i] with B[j], so skip this toy if we still can
            if not skip_used:
                skip_used = True
                unmatched_toy_index = i
                i += 1
            else:
                # Already skipped one toy before, so it's impossible now
                print(-1)
                return

    # If we haven't processed all toys, some are left over with no boxes left
    if i < N:
        leftover = N - i
        if skip_used:
            # We have no more skips available
            # If there's even 1 leftover toy, it's impossible
            if leftover > 0:
                print(-1)
                return
        else:
            # We still have 1 skip available
            # We must skip exactly one leftover toy if leftover == 1
            if leftover == 1:
                skip_used = True
                unmatched_toy_index = i
                i += 1
            else:
                # More than one leftover toy means we can't fix with one skip
                print(-1)
                return

    # Number of matched toys is how many boxes we used
    matched = j  # we increment j each time we match a toy

    # We need exactly N-1 matches, having used exactly one skip
    if matched == (N - 1) and skip_used:
        print(A[unmatched_toy_index])
    else:
        print(-1)

# Do not forget to call main()
if __name__ == "__main__":
    main()