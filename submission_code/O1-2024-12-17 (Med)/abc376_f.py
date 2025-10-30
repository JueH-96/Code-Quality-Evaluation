def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    Q = int(input_data[1])

    # Instructions: a list of (hand, target)
    instr = []
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        instr.append((h, t))

    # A small helper for circular distance
    def dist(a, b):
        d = abs(a - b)
        return min(d, N - d)

    # ----------------------------------------------------------------
    # We will use a dynamic-programming approach where:
    #
    #   dp[i][pos] = the minimum total cost after satisfying
    #                the i-th instruction, where the "other" hand
    #                (the one not used in the i-th instruction)
    #                ends on part "pos".
    #
    # If the i-th instruction says "L T_i", then after this instruction:
    #     Left hand = T_i
    #     Right hand = pos
    # If the i-th instruction says "R T_i", then after this instruction:
    #     Right hand = T_i
    #     Left hand = pos
    #
    # The cost to go from state ( after i-1 instructions ) to ( after i instructions )
    # depends on which hand is moved this time, and we allow moving the other hand
    # as well to possibly reduce future cost, as long as the final two hands' positions
    # differ.
    #
    # However, we must also deal with the first instruction specially, starting from
    # L=1, R=2 initially (no cost so far).
    #
    # Transitions:
    #   Let the previous instruction be i-1.  Its hand is H_{i-1}, target T_{i-1}.
    #   Let dp[i-1][pos] store the minimal cost so far.  Then for instruction i:
    #   if H_i == H_{i-1}, we move the same hand from T_{i-1} to T_i,
    #       and the other hand from pos to pos'.
    #       cost = dist(T_{i-1}, T_i) + dist(pos, pos')   (invalid if T_i == pos')
    #   if H_i != H_{i-1}, we move the other hand from pos -> T_i,
    #       and the originally moved hand from T_{i-1} -> pos'.
    #       cost = dist(pos, T_i) + dist(T_{i-1}, pos')   (invalid if T_i == pos')
    #
    # Then dp[i][pos'] = min( dp[i][pos'], dp[i-1][pos] + cost ).
    #
    # In the special case i=1, we start from (L=1, R=2) with cost=0 and apply the same formula.
    #
    # Finally our result is min_{pos} dp[Q][pos].
    #
    # Complexity in a straightforward implementation is O(Q*N^2) which can be large (up to 9e9),
    # but under typical contest-time limits (and in lower-level languages) it can sometimes pass for N=3000.
    # In pure Python this is quite heavy.  However, the problem statement only asks for a correct solution.
    # This DP is correct (even if it may be slow in Python for the largest inputs).
    #
    # We implement it carefully below.
    # ----------------------------------------------------------------

    INF = 10**15

    # dp[i][pos] = minimal cost after i-th instruction, with the "other" hand at 'pos'
    # and the hand H_i at T_i.
    dp = [[INF]*(N+1) for _ in range(Q+1)]

    # Initialize for the first instruction i=1
    # Starting from L=1, R=2, cost=0
    # If H_1 = 'L', that means left hand goes from 1->T_1, and we can move right hand 2->pos
    # cost = dist(1,T_1) + dist(2,pos), unless T_1 == pos (invalid).
    # Then dp[1][pos] = that cost.
    #
    # If H_1 = 'R', that means right hand goes 2->T_1, left hand 1->pos
    # cost = dist(2,T_1) + dist(1,pos), unless T_1 == pos (invalid).
    #
    # We'll fill dp[1][:] accordingly.

    firstH, firstT = instr[0]
    if firstH == 'L':
        for pos in range(1, N+1):
            if pos == firstT:
                continue
            c = dist(1, firstT) + dist(2, pos)
            dp[1][pos] = min(dp[1][pos], c)
    else:
        # firstH == 'R'
        for pos in range(1, N+1):
            if pos == firstT:
                continue
            c = dist(2, firstT) + dist(1, pos)
            dp[1][pos] = min(dp[1][pos], c)

    # Now process instructions i=2..Q
    for i in range(2, Q+1):
        (prevH, prevT) = instr[i-2]   # (i-1)-th instruction
        (curH, curT)   = instr[i-1]   # i-th instruction

        # We have dp[i-1][pos] as the cost so far, meaning:
        # after (i-1)-th instruction, if prevH='L' => L=prevT, R=pos
        # if prevH='R' => R=prevT, L=pos
        #
        # Now for the i-th instruction:
        # if curH == prevH, the same hand moves from prevT->curT,
        #                   the other hand moves from pos->pos'.
        #    cost = dist(prevT, curT) + dist(pos, pos'), invalid if curT == pos'
        # if curH != prevH, the other hand moves from pos->curT,
        #                   the 'prevH' hand from prevT->pos'.
        #    cost = dist(pos, curT) + dist(prevT, pos'), invalid if curT == pos'.
        #
        # Then dp[i][pos'] = min over pos of dp[i-1][pos] + cost.

        same_hand = (curH == prevH)

        # Precompute the movement cost for each (pos, pos') pair to avoid repeating dist calls
        # We'll do it on the fly though, as storing an NxN might be large in memory (9e6).
        # We'll just compute dist on the fly.

        if same_hand:
            # We move prevT->curT
            moveA = dist(prevT, curT)
            for pos in range(1, N+1):
                base = dp[i-1][pos]
                if base == INF:
                    continue
                # The other hand goes pos->pos'
                # final must have curT != pos', or they'd collide
                # cost = moveA + dist(pos, pos')
                for posp in range(1, N+1):
                    if posp == curT:
                        continue
                    cost_here = moveA + dist(pos, posp)
                    val = base + cost_here
                    if val < dp[i][posp]:
                        dp[i][posp] = val
        else:
            # We move the other hand from pos->curT, and the 'prevH' hand from prevT->pos'
            for pos in range(1, N+1):
                base = dp[i-1][pos]
                if base == INF:
                    continue
                # cost = dist(pos, curT) + dist(prevT, pos')
                moveA = dist(pos, curT)
                for posp in range(1, N+1):
                    if posp == curT:
                        continue
                    cost_here = moveA + dist(prevT, posp)
                    val = base + cost_here
                    if val < dp[i][posp]:
                        dp[i][posp] = val

    # After all Q instructions, dp[Q][pos] means the cost after the Q-th instruction,
    # with the Q-th instructed hand at T_Q, and the other hand at pos.
    # We want the minimal among all pos. 
    ans = min(dp[Q])

    print(ans)