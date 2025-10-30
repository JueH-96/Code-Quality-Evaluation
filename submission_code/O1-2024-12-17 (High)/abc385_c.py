def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    if N == 1:
        # If there's only one building, the answer is always 1
        print(1)
        return

    # Read the heights
    H = list(map(int, input_data[1:]))

    from collections import defaultdict

    # Group building indices by their heights
    height_dict = defaultdict(list)
    for i, h in enumerate(H):
        height_dict[h].append(i)

    global_max = 1

    # For each distinct height, find the longest arithmetic progression of indices
    for h, positions in height_dict.items():
        M = len(positions)
        # If this group's size can't exceed what we already have, skip
        if M <= global_max:
            continue
        # If M == 2, we can immediately set global_max = 2 if it's better
        if M == 2:
            global_max = max(global_max, 2)
            continue

        # Standard "longest arithmetic progression" DP
        # dp[j] will be a dict: difference -> length of AP ending at j with step 'difference'
        dp = [dict() for _ in range(M)]
        local_max = 1  # At least 1 element

        for j in range(M):
            pj = positions[j]
            dpj = dp[j]
            for i in range(j):
                pi = positions[i]
                d = pj - pi
                dpi = dp[i]
                # If there's an AP ending at i with difference d, extend it; otherwise start new with length 2
                length = dpi.get(d, 1) + 1
                dpj[d] = length
                if length > local_max:
                    local_max = length

        # Update global maximum
        if local_max > global_max:
            global_max = local_max

    print(global_max)

# Do not forget to call the main function
if __name__ == "__main__":
    main()