def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    M = int(input_data[0])
    S1 = input_data[1]
    S2 = input_data[2]
    S3 = input_data[3]
    
    # For convenience, store the reel strings in a list
    reels = [S1, S2, S3]
    
    # Precompute, for each reel i and each digit d in '0'..'9',
    # the sorted list of times (t) up to about 3*M+5 such that
    # S_i[t % M] == d.  In other words, if S_i[pos] == d, then
    # times t = pos + k*M for k=0..3 (or 4) as long as t <= 3*M+5.
    # We store these in arrays times_for_digit[i][d].
    
    # First, find for each reel i and digit d, the positions pos in [0..M-1] where S_i[pos] == d.
    positions = [[[] for _ in range(10)] for _ in range(3)]
    for i in range(3):
        for pos in range(M):
            d = int(reels[i][pos])
            positions[i][d].append(pos)
    
    # Now build the time lists
    # We'll go up to 3*M + 5 to be safe.
    limit = 3 * M + 5
    times_for_digit = [[[] for _ in range(10)] for _ in range(3)]
    
    for i in range(3):
        for d in range(10):
            if positions[i][d]:  # only if that digit appears in reel i
                # For each position p where this digit appears
                # add p + k*M for k such that it does not exceed limit
                for p in positions[i][d]:
                    k = 0
                    while True:
                        t = p + k * M
                        if t > limit:
                            break
                        times_for_digit[i][d].append(t)
                        k += 1
                # sort the list of times
                times_for_digit[i][d].sort()
    
    import math
    ans = math.inf
    
    # We will try each digit d in 0..9.
    # For each d, if all reels can show d (i.e. times_for_digit[i][d] is non-empty),
    # we look for three distinct press times t1, t2, t3 (one from each reel),
    # minimizing max(t1,t2,t3).
    #
    # A direct triple nested loop can be too large. Instead, we do:
    #  for each pair (a in A1, b in A2) with a != b:
    #    let X = max(a, b)
    #    we check whether there's a c in A3 that is distinct from a, b
    #    that can make max(a,b,c) as small as possible.
    #
    #    The best "c" to keep T = max(a,b,c) minimal is:
    #      - If there's a c < X and c != a, b, then T = X
    #        (since max(a,b,c) = X in that case).
    #      - Otherwise, we pick the smallest c >= X that isn't a or b,
    #        then T = that c if c > X, or T = X if c == X (and X not in {a,b}).
    #
    # We'll do a binary search in A3 to find positions around X.
    
    import bisect
    
    for d in range(10):
        # Check if all three reels can show digit d
        if not times_for_digit[0][d]:
            continue
        if not times_for_digit[1][d]:
            continue
        if not times_for_digit[2][d]:
            continue
        
        A1 = times_for_digit[0][d]
        A2 = times_for_digit[1][d]
        A3 = times_for_digit[2][d]
        
        # We'll iterate over all a in A1, b in A2 (with a != b)
        # Then do a binary search in A3 for c.
        # We'll keep track of the minimal possible max(a,b,c).
        
        # We can stop searching once max(a,b) already exceeds any best-known answer,
        # as we can't do better than that.
        curr_min = math.inf
        
        j_len = len(A2)
        k_len = len(A3)
        
        for a in A1:
            # If a already beyond current best, no need to proceed
            if a > curr_min:
                break
            for b in A2:
                if b == a:
                    continue
                X = max(a, b)
                # If X already > curr_min, break
                if X > curr_min:
                    # Because b is only going to get bigger or smaller in no predictable order,
                    # we can't strictly break from the b-loop always.
                    # But typically if b≥a and X≥curr_min, the next b might be smaller or bigger.
                    # We'll just check and continue if we can't do better.
                    continue
                
                # We'll do a binary search in A3 around X.
                # 1) We see if there's any c < X not equal to a or b.
                pos = bisect.bisect_left(A3, X)
                
                # Check c just below pos, descending.
                found_better = False
                idx = pos - 1
                while idx >= 0:
                    c = A3[idx]
                    if c != a and c != b:
                        # Then T = X
                        if X < curr_min:
                            curr_min = X
                        found_better = True
                        break
                    idx -= 1
                
                if found_better:
                    # We found c < X, so T = X is the minimal for this (a,b).
                    # We can't do better than X for these exact a,b, so let's proceed to next pair.
                    continue
                
                # 2) If not found c < X, pick the smallest c >= X that is not a or b.
                idx = pos
                while idx < k_len:
                    c = A3[idx]
                    if c != a and c != b:
                        T = max(X, c)
                        if T < curr_min:
                            curr_min = T
                        break
                    idx += 1
                
                # If idx == k_len, we found no c >= X that is distinct => no solution from those.
        
        if curr_min < ans:
            ans = curr_min
    
    if ans == math.inf:
        print(-1)
    else:
        print(ans)

# Don't forget to call main()!
if __name__ == "__main__":
    main()