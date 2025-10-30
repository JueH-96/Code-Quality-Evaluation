def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    heights = list(map(int, data[1:1+n]))
    
    # Group indices by building height. Indices are stored by 1-indexing.
    groups = {}
    for i, h in enumerate(heights, start=1):
        groups.setdefault(h, []).append(i)
    
    best = 1  # At least one building can always be chosen.
    
    # For each group (same height), we need to select indices that form an arithmetic progression.
    # We'll use a DP approach for Longest Arithmetic Progression (LAP) on the sorted list of indices.
    for h, pos in groups.items():
        L = len(pos)
        # If the group size is not bigger than current best, skip (since even if all were AP, count <= best)
        if L <= best:
            continue
        if L == 1:
            continue  # Only one building, which is already considered.
        
        # dp[j] is a dictionary mapping a common difference d
        # to the length of the AP ending at index position pos[j].
        dp = [dict() for _ in range(L)]
        local_max = 1
        for j in range(L):
            for i in range(j):
                d = pos[j] - pos[i]
                # If an AP ending at pos[i] with difference d exists, extend it, or start new with length 2.
                curr = dp[i].get(d, 1) + 1
                dp[j][d] = curr
                if curr > local_max:
                    local_max = curr
        if local_max > best:
            best = local_max

    sys.stdout.write(str(best))

if __name__ == '__main__':
    main()