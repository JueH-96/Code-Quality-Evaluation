def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    H = list(map(int, data[1:1+n]))
    
    # For each building height, we store a list of its indices.
    # (Since the buildings are given in order the indices will come out in increasing order.)
    max_h = max(H)
    groups = [[] for _ in range(max_h + 1)]
    for i, h in enumerate(H):
        groups[h].append(i)
        
    # Global answer: if we pick one building we get 1.
    res = 1
    
    # For each group (each fixed height), we want to obtain the maximum number of indices
    # that can form an arithmetic progression.
    # Notice: if the entire index list is already arithmetic (i.e. consecutive difference is constant),
    # then the answer for this height is simply the group’s size.
    # Otherwise, we use a DP solution (dictionary method) to compute the length.
    for grp in groups:
        m = len(grp)
        if m == 0:
            continue
        # Even if you could take all indices from this group,
        # if its size does not beat our current best, we can skip.
        if m <= res:
            continue
        if m < 3:
            if m > res:
                res = m
            continue
        
        # Check quickly if the entire list is an arithmetic progression.
        d = grp[1] - grp[0]
        is_arith = True
        for k in range(2, m):
            if grp[k] - grp[k-1] != d:
                is_arith = False
                break
        if is_arith:
            if m > res:
                res = m
            continue

        # Use a dynamic programming approach.
        # dp[j] will be a dictionary where for each difference d
        # dp[j][d] is the length of an arithmetic progression ending at grp[j] with difference d.
        dp = [{} for _ in range(m)]
        local_max = 2  # any two indices form an AP.
        P = grp  # alias for clarity
        for j in range(1, m):
            pj = P[j]
            dpj = dp[j]
            for i in range(j):
                diff = pj - P[i]
                # By default a single element gives length 1.
                # When we pair with a previous index we get length = dp[i].get(diff, 1) + 1.
                cand = dp[i].get(diff, 1) + 1
                # Set dp[j][diff] so that if multiple i's give the same diff, we store the longest chain.
                if cand > dpj.get(diff, 0):
                    dpj[diff] = cand
                if cand > local_max:
                    local_max = cand
            # End inner loop for j.
        if local_max > res:
            res = local_max
    
    sys.stdout.write(str(res))
    
if __name__ == '__main__':
    main()