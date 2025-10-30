def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = data[1].strip()
    # Build an array of dp pairs for the leaves.
    # For a leaf whose value is '0', no change is needed to have 0 so cost0 = 0 and
    # changing it to 1 costs 1, so dp = (0, 1).
    # Similarly, for '1', dp = (1, 0).
    dp = []
    for ch in A:
        if ch == '0':
            dp.append((0, 1))
        else:
            dp.append((1, 0))
    
    # The tree is built by combining groups of 3 from the current level.
    # For an internal node with children with dp values (a0,a1), (b0,b1), (c0,c1):
    # Its value is the majority of 3 children. To force the parent's value to 0,
    # we need at least two children equal to 0.
    # So, we consider two possibilities:
    #  1) Force all children to 0: cost = a0 + b0 + c0.
    #  2) Force exactly two children to 0 and one remains 1.
    #     There are 3 choices:
    #      - Force child1 to flip to 1, others to 0: cost = a1 + b0 + c0.
    #      - Force child2 to flip to 1, others to 0: cost = a0 + b1 + c0.
    #      - Force child3 to flip to 1, others to 0: cost = a0 + b0 + c1.
    # The parent's minimal cost for 0 is the minimum of these two options.
    #
    # Similarly, to force the parent's value to 1, we have:
    #  1) Force all children to 1: cost = a1 + b1 + c1.
    #  2) Force exactly two children to 1 and one to 0:
    #      - Option 1: a0 + b1 + c1.
    #      - Option 2: a1 + b0 + c1.
    #      - Option 3: a1 + b1 + c0.
    #
    # We then update dp array level by level.
    while len(dp) > 1:
        new_dp = [None] * (len(dp) // 3)
        idx = 0
        for i in range(0, len(dp), 3):
            a0, a1 = dp[i]
            b0, b1 = dp[i + 1]
            c0, c1 = dp[i + 2]
            
            # For parent's value 0:
            cost_all0 = a0 + b0 + c0
            cost_two0 = a1 + b0 + c0
            t = a0 + b1 + c0
            if t < cost_two0:
                cost_two0 = t
            t = a0 + b0 + c1
            if t < cost_two0:
                cost_two0 = t
            parent0 = cost_all0 if cost_all0 < cost_two0 else cost_two0
            
            # For parent's value 1:
            cost_all1 = a1 + b1 + c1
            cost_two1 = a0 + b1 + c1
            t = a1 + b0 + c1
            if t < cost_two1:
                cost_two1 = t
            t = a1 + b1 + c0
            if t < cost_two1:
                cost_two1 = t
            parent1 = cost_all1 if cost_all1 < cost_two1 else cost_two1
            
            new_dp[idx] = (parent0, parent1)
            idx += 1
        dp = new_dp

    # At the root dp[0], the dp pair (cost_forcing0, cost_forcing1) tells us the minimum
    # number of leaf changes to force a given final bit.
    # Notice that if we do no changes, the original computed value gets cost 0.
    # So if dp[0][0] == 0 the original final bit is 0; otherwise, it is 1.
    root0, root1 = dp[0]
    if root0 == 0:
        # Original final bit is 0; we must flip it to 1.
        ans = root1
    else:
        # Original final bit is 1; we must flip it to 0.
        ans = root0
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()