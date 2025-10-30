def main():
    import sys
    from collections import defaultdict

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    H = list(map(int, data[1:]))

    # Edge case: if there is only 1 building, answer is 1
    if N == 1:
        print(1)
        return

    # Group indices by building height
    groups = defaultdict(list)
    for i, h in enumerate(H):
        groups[h].append(i)

    ans = 1  # At least one building can always be chosen

    # For each distinct height, find the longest arithmetic subsequence of indices
    for height, idx_list in groups.items():
        M = len(idx_list)
        # If we can't improve on the current answer with this height, skip
        if M <= ans:
            ans = max(ans, M)
            continue
        # If M == 2, the longest AP for these two indices is 2
        if M == 2:
            ans = max(ans, 2)
            continue

        # DP to find the longest arithmetic progression among indices in idx_list
        dp = [defaultdict(int) for _ in range(M)]
        local_max = 2  # We know at least a pair forms length = 2
        for j in range(M):
            for i in range(j):
                d = idx_list[j] - idx_list[i]
                # If dp[i][d] doesn't exist, it means we had length=1 at i (by itself),
                # so adding j forms length=2. Else extend the known progression.
                length = dp[i].get(d, 1) + 1
                dp[j][d] = length
                if length > local_max:
                    local_max = length
            # Early exit if we've reached the maximum possible for this height
            if local_max == M:
                break

        ans = max(ans, local_max)

    print(ans)

# Don't forget to call main()
main()