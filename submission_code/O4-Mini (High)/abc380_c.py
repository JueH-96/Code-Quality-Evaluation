def main():
    import sys
    input = sys.stdin.readline

    # Read inputs
    N, K = map(int, input().split())
    S = input().rstrip()

    # Find the start and end positions of each 1-block
    starts = []
    ends = []
    for i, ch in enumerate(S):
        if ch == '1' and (i == 0 or S[i-1] == '0'):
            starts.append(i)
        if ch == '1' and (i == N-1 or S[i+1] == '0'):
            ends.append(i)

    # We are guaranteed there are at least K blocks
    # Indexing in Python lists is 0-based; blocks are 1-based in problem
    prev_idx = K - 2
    curr_idx = K - 1

    r_prev = ends[prev_idx]
    l_curr = starts[curr_idx]
    r_curr = ends[curr_idx]

    # Build the resulting string:
    # 1) Prefix up to r_prev
    # 2) The moved block (all 1's from l_curr to r_curr)
    # 3) The zeros that were between the two blocks
    # 4) The suffix after r_curr
    prefix = S[:r_prev+1]
    moved_block = S[l_curr:r_curr+1]
    zeros_between = '0' * (l_curr - (r_prev + 1))
    suffix = S[r_curr+1:]

    result = prefix + moved_block + zeros_between + suffix

    # Output the answer
    sys.stdout.write(result)

# Call main to execute
if __name__ == "__main__":
    main()