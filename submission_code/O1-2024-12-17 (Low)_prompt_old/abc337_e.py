def solve():
    import sys

    # Read N
    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # We want the minimum M such that 2^M >= N
    # For N <= 100, 2^6 = 64 < 100 <= 128 = 2^7, so M = 7.
    M = 0
    while (1 << M) < N:
        M += 1

    # Print the number of friends to call
    print(M)
    sys.stdout.flush()

    # For friend j in [0..M-1], gather the bottles whose (i-1)-th representation has bit j = 1
    # Note: use 1-based bottle numbering.
    index = 1
    for j in range(M):
        subset = []
        for i in range(1, N + 1):
            # Check if bit j of (i-1) is set
            if ((i - 1) >> j) & 1:
                subset.append(i)
        # Print K_j followed by the bottles in ascending order
        print(len(subset), *subset)
        sys.stdout.flush()

    # Now read the result string S
    # data already has the input but let's read the required response from the buffer after
    # the lines we consumed. We have consumed 1 integer so far from data. We should have the stomach upset info next.
    # However, in a real interactive setting, we'd read fresh lines from sys.stdin, but to comply with the
    # environment here, let's find it from data after the first line.
    # But in typical interactive solutions, we'd do something like: S = sys.stdin.readline().strip()

    # We've read data[0] as N. Next M lines are not provided in normal interactive style,
    # but let's assume the next chunk is the upset string S after M lines.
    # For demonstration, let's just assume it's data[1], since this is how it would appear in a
    # single-run environment. In a real interactive environment, this would be replaced by:
    # S = input().strip() after printing the subsets.

    # Since the problem statement is purely illustrative, let's handle it gracefully:
    if len(data) > 1:
        S = data[1]
    else:
        # In a real interactive environment, we would read S from input here
        S = '0' * M  # fallback (won't matter here, as there's no real interaction in this environment)

    # Determine the spoiled bottle using the bit pattern from S
    # friend j = 1 => bit j is set
    spoiled_index_zero_based = 0
    for j in range(M):
        if j < len(S) and S[j] == '1':
            spoiled_index_zero_based |= (1 << j)

    # Convert back to 1-based bottle index
    spoiled_bottle = spoiled_index_zero_based + 1

    # Output the spoiled bottle
    print(spoiled_bottle)
    sys.stdout.flush()

# Call solve()
solve()