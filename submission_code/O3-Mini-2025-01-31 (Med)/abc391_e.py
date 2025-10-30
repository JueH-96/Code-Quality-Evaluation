def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = data[1].strip()
    
    # First, simulate the majority operations to compute the final bit without any changes.
    current = [1 if ch == '1' else 0 for ch in A]
    while len(current) > 1:
        nxt = []
        for i in range(0, len(current), 3):
            # For each group of three, the majority (at least 2 ones gives a 1)
            a, b, c = current[i], current[i+1], current[i+2]
            nxt.append(1 if (a + b + c) >= 2 else 0)
        current = nxt
    orig_final = current[0]

    # We now build a dynamic programming solution bottom up on the tree of operations.
    # At the leaf level, the dp for each leaf is:
    #   dp_leaf[0] = 0 if the bit is 0 else 1 (one change needed to turn a 1 into a 0)
    #   dp_leaf[1] = 0 if the bit is 1 else 1 (one change needed to turn a 0 into a 1)
    dp = []
    for ch in A:
        if ch == '0':
            dp.append((0, 1))
        else:
            dp.append((1, 0))
    
    # In the internal nodes, each node covers three children.
    # The parent's value is defined as the majority of its children.
    # To force the parent's value to be t (0 or 1), we need at least two of its children to be t.
    # For the third child we can permit it to be either t or not t.
    # So for each triple (child1, child2, child3), we compute:
    #   Option1: Force all three children to t.
    #   Option2: Force exactly two children to t and let the remaining child be free 
    #            (free means we take the minimal cost among forcing it to 0 or 1).
    # We choose the option with the minimal total changes.
    while len(dp) > 1:
        new_dp = []
        for i in range(0, len(dp), 3):
            d1, d2, d3 = dp[i], dp[i+1], dp[i+2]
            new_val = [None, None]  # new_val[0] and new_val[1] will store the minimal cost to force 0 or 1 respectively.
            for t in (0, 1):
                # Option 1: All three children forced to t.
                cost_all = d1[t] + d2[t] + d3[t]
                # Option 2: Force exactly two of them to t and let the other be free.
                # "Free" means we can choose the child's value with minimum fewer changes.
                free_d1 = min(d1[0], d1[1])
                free_d2 = min(d2[0], d2[1])
                free_d3 = min(d3[0], d3[1])
                cost_opt1 = d1[t] + d2[t] + free_d3
                cost_opt2 = d1[t] + d3[t] + free_d2
                cost_opt3 = d2[t] + d3[t] + free_d1
                best = cost_all
                if cost_opt1 < best:
                    best = cost_opt1
                if cost_opt2 < best:
                    best = cost_opt2
                if cost_opt3 < best:
                    best = cost_opt3
                new_val[t] = best
            new_dp.append((new_val[0], new_val[1]))
        dp = new_dp

    # Our final dp[0] covers the entire tree (A').
    # To change the final computed bit from orig_final to the opposite (1 - orig_final),
    # we need the cost dp[0][1 - orig_final].
    answer = dp[0][1 - orig_final]
    sys.stdout.write(str(answer))

if __name__ == "__main__":
    main()