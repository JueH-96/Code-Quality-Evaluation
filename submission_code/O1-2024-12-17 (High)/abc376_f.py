def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    # Instructions: H_i, T_i
    # H_i in {'L','R'}, T_i in [1..N]
    instructions = input_data[2:]
    
    # ------------------------------------------------------------
    # 1) We will solve this using a state-based BFS for each instruction.
    #
    #    State representation on the ring:
    #
    #    - We label the ring parts 0..(N-1) internally (0-based) rather
    #      than 1..N. Part i and i+1 (mod N) are adjacent.
    #
    #    - We track the positions of (L, R).  They must always differ.
    #
    #    However, storing dp[i][L][R] is too large (Q*N*N ~ 9e9).
    #    Instead, note that after instruction i, exactly one hand
    #    is at T_i, and the "other" hand could be at any of the other
    #    N-1 positions.  We only need to store dp[i][pos_of_other_hand].
    #
    #    Then to transition from dp[i-1][pos'] to dp[i][pos], we do:
    #       - Figure out what (L_{i-1}, R_{i-1}) was.
    #       - BFS in the (L,R) graph from that one state to every state
    #         that satisfies the i-th instruction (namely L = T_i if H_i='L',
    #         or R = T_i if H_i='R').  Among those "final" states, we record
    #         the position of the other hand and update dp[i][that_position].
    #
    #    But we have multiple "start" states (up to N of them) from dp[i-1].
    #    We do a multi-source BFS: we enqueue all (L_{i-1},R_{i-1}) states
    #    found from dp[i-1] with the appropriate initial cost.  Then we
    #    spread out in the no-collision ring graph.  Whenever we reach a
    #    state that satisfies the i-th instruction, we update dp[i].
    #
    # 2) Encoding states for BFS:
    #
    #    Because L != R, we can represent a state by (L, d) where
    #        d = (R - L) mod N,  d in [1..N-1].
    #    So R = (L + d) mod N.
    #    The number of such pairs is N*(N-1).
    #
    # 3) Adjacency:
    #    From (L, d), let R = (L+d)%N.
    #    Moves (cost = 1 each):
    #       - Move L forward by 1 if L+1 != R  => that requires d != 1
    #         new L' = (L+1) mod N
    #         new d' = (R - L') mod N = ( (L+d) - (L+1) ) mod N = d - 1
    #       - Move L backward by 1 if L-1 != R => that requires d != N-1
    #         new L' = (L-1) mod N
    #         new d' = (R - L') mod N = ( (L+d) - (L-1) ) mod N = d + 1
    #         (all mod N, but d in [1..N-1], so d+1 means (d+1) in [2..N], must not be 0 => d != N-1)
    #       - Move R forward by 1 if (R+1)!=L => that requires d != N-1
    #         R' = (R+1) mod N => new d' = R' - L (mod N) = ( (L+d)+1 - L ) mod N = d+1
    #       - Move R backward by 1 if (R-1)!=L => that requires d != 1
    #         R' = (R-1) mod N => new d' = (R'-L) mod N = ( (L+d)-1 - L ) mod N = d-1
    #
    # 4) BFS approach per instruction:
    #
    #    - We have dp[i-1][pos'] (the minimal cost to end instruction i-1
    #      with the hand H_{i-1} at T_{i-1} and the "other" at pos').
    #    - We decode that into exactly one (L,R) = (T_{i-1}, pos') or (pos', T_{i-1}),
    #      depending on H_{i-1}.  Then we encode it as (L, d) or (pos', d).
    #    - We do a multi-source BFS: dist[start_state] = dp[i-1][pos'].
    #      Put all start_states in a queue with their initial dist.
    #
    #    - Then for each popped state (L,d), cost=dist[L,d].
    #        if H_i='L' and L == T_i => we've satisfied instruction i
    #            => otherHand = R = (L+d)%N
    #            => dp[i][otherHand] = min of dp[i][otherHand], cost
    #        if H_i='R' and R == T_i => we've satisfied instruction i
    #            => otherHand = L
    #            => dp[i][otherHand] = min of dp[i][otherHand], cost
    #      Then expand neighbors with cost+1 if not visited.
    #
    #    - We can stop BFS early if we've found all possible "otherHand"
    #      positions (N-1 of them) for that instruction.  (We do not need
    #      "otherHand" = T_i, because L!=R always.)
    #
    # 5) Finally, the answer is min(dp[Q][pos]) over all pos in [0..N-1].
    #
    # 6) Complexity considerations:
    #    - Each BFS can visit up to N*(N-1) ~ 9e6 states for N=3000.
    #    - Q can be up to 3000, so worst-case ~2.7e10 operations, which is
    #      very large for Python.  In practice, an optimized C++ solution might
    #      handle it if carefully done and if the test data isn’t maxed out.
    #
    #    We will implement the described solution (it is the cleanest correct
    #    method).  In slower Python environments on worst-case data it might be
    #    too slow, but this is the standard approach.
    #
    # ------------------------------------------------------------

    # Convert hand labels 'L'/'R' into 0/1 to make it simpler.
    usedHand = [0]*Q  # 0 for L, 1 for R
    Tarray   = [0]*Q
    idx = 0
    for i in range(Q):
        h = instructions[idx]
        idx += 1
        t = int(instructions[idx]); idx += 1
        usedHand[i] = 0 if h == 'L' else 1
        # convert t from [1..N] to [0..N-1]
        Tarray[i] = t - 1

    INF = 10**15

    # dp[i][pos] = minimum cost after i instructions,
    #              if after the i-th instruction the "other" hand is at "pos".
    # i.e. if usedHand[i-1] = 0 => L = Tarray[i-1], R = pos
    #      if usedHand[i-1] = 1 => R = Tarray[i-1], L = pos
    dp = [ [INF]*N for _ in range(Q+1) ]

    # Initial condition: before any instructions (i=0), we have L=0, R=1 (in 0-based),
    # cost = 0, and we pretend "the last used hand was L" so that dp[0][R=1] = 0
    # is consistent.  This aligns with "Initially, left hand on part 1 (0-based=0),
    # right hand on part 2 (0-based=1)."
    # We'll store dp[0][1] = 0, meaning "if the last used hand was L => L=0, R=1 => cost=0."
    # That sets up for the 1st instruction.

    # However, if the first instruction is "R something", we can still start
    # from L=0,R=1 by reading dp[0][some_pos], decoding it as if 'L' was used
    # at i=0.  That won't break anything.  BFS from that state will just work.
    dp[0][1] = 0

    # Precompute neighbors function in code for quick checks:
    # We'll just do them inline to avoid function call overhead.

    from collections import deque

    for i in range(1, Q+1):
        h = usedHand[i-1]   # which hand is used in the (i)-th instruction (index i-1)
        t = Tarray[i-1]     # target part (0-based)

        # We'll do a BFS over (L,d) states with multi-source from dp[i-1].
        dist = [-1]*(N*(N-1))  # -1 means unvisited

        # queue
        q = deque()

        # Prepare multi-source:
        # For each possible "otherPos" in [0..N-1]:
        #   costPrev = dp[i-1][otherPos]
        #   if costPrev < INF:
        #     decode the previous (L,R).
        #     if usedHand[i-1] == 0 => L= t_{i-1}, R= otherPos
        #     else => R= t_{i-1}, L= otherPos
        #     encode into (L,d)
        #     dist[...] = costPrev
        #     push
        costPrevArr = dp[i-1]  # for convenience

        for otherPos in range(N):
            baseCost = costPrevArr[otherPos]
            if baseCost == INF:
                continue
            # decode (L,R)
            if h == 0:
                # hand used last time was L => L_{i-1}= t => R_{i-1}= otherPos
                # but that h is for the i-th instruction. We want i-1's usedHand:
                # usedHand[i-1] is h for the i-th instruction, not i-1th.
                # Careful: we want the hand used in instruction (i-1).
                # Actually, the array "usedHand" at index (i-1) is for the i-th instruction
                # in 1-based sense. We want the (i-1)-th instruction's hand => that's usedHand[i-2]
                # This is confusing. Let's clarify:
                #
                # dp[i][pos] depends on dp[i-1], but the hand used in "instruction i-1"
                # is usedHand[i-2]. But here "i" in code is 1..Q => "i-1" in code is 0..Q-1
                # so usedHand[i-1] is actually the hand for the i-th instruction (1-based).
                #
                # But to decode the state from dp[i-1], we actually need usedHand[i-2].
                # At i=1, that index is -1 => there's no real instruction i-0 => the base case.
                #
                # To handle this cleanly, we store "which hand was used to get dp[i]" is usedHand[i-1].
                # So "dp[i]" means after i instructions, the i-th instruction used usedHand[i-1].
                # So if we want to decode dp[i-1], the hand used is usedHand[i-2].
                #
                # But for i=1, dp[i-1]=dp[0], that was the base condition with L=0,R=1.
                #
                # We'll do it this way: when we do BFS for the i-th instruction,
                # the "start states" are given by dp[i-1], and the hand used in the (i-1)-th
                # instruction is usedHand[i-2] if i>1. If i=1, we do the base condition special.
                #
                # We'll define a small helper function to decode the positions from dp[i-1].
                pass

        # Actually, to decode the correct hand from dp[i-1], we need usedHand for (i-1)th instruction => usedHand[i-2].
        # But if i==1, we are coming from the base state (L=0,R=1).
        # We'll define a small function:

        def decode_positions(i_minus_1, otherPos):
            """
            Returns (L0,R0) given that we have just finished instruction (i_minus_1),
            and the 'other hand' ended at otherPos with minimal cost dp[i_minus_1][otherPos].
            The hand used in that instruction is usedHand[i_minus_2].
            If i_minus_1=0, it's the base condition (L=0,R=1).
            """
            if i_minus_1 == 0:
                # base condition
                # we said dp[0][1]=0 means L=0,R=1 was how we ended "0 instructions"
                # ignoring an actual instruction. So that is (L,R)=(0,1).
                # But that only is valid if otherPos=1. If it's something else, dp[0][x] should be INF.
                return (0,1)
            else:
                # the hand used in instruction i_minus_1 is usedHand[i_minus_2]
                hand_used_prev = usedHand[i_minus_1-1]
                targ_prev = Tarray[i_minus_1-1]
                if hand_used_prev == 0:
                    # L was used => L = targ_prev, R = otherPos
                    return (targ_prev, otherPos)
                else:
                    # R was used => R = targ_prev, L = otherPos
                    return (otherPos, targ_prev)

        # Build the list of start states:
        starts = []
        for otherPos in range(N):
            cost0 = dp[i-1][otherPos]
            if cost0 == INF:
                continue
            (L0, R0) = decode_positions(i-1, otherPos)
            # encode L0, R0 => (L0, d0)
            d0 = (R0 - L0) % N
            if d0 == 0:
                # invalid but shouldn't happen if L0!=R0
                continue
            start_index = L0*(N-1)+(d0-1)
            dist[start_index] = cost0
            starts.append(start_index)

        # We'll BFS from these starts.
        queue = deque(starts)

        newDist = [INF]*N  # dp[i][pos], we want min cost of finishing instruction i with "other hand"=pos
        found_count = 0    # how many distinct "otherPos" we've found a cost for

        # BFS
        while queue:
            u = queue.popleft()
            cost_u = dist[u]
            L = u // (N-1)
            d = (u % (N-1)) + 1
            R = (L + d) % N

            # check if (L,R) satisfies the i-th instruction:
            # usedHand for i-th instruction is usedHand[i-1], let's call it hThis = usedHand[i-1].
            # if hThis=0 => L must be Tarray[i-1], if L==t => we've satisfied this instruction
            # if hThis=1 => R must be t => ...
            if h == 0:
                # L must be t
                if L == t:
                    # then other hand is R
                    if cost_u < newDist[R]:
                        newDist[R] = cost_u
                        if newDist[R] == cost_u and cost_u < INF:  # it was improved
                            found_count += 1
            else:
                if R == t:
                    # then other hand is L
                    if cost_u < newDist[L]:
                        newDist[L] = cost_u
                        if newDist[L] == cost_u and cost_u < INF:  # it was improved
                            found_count += 1

            if found_count >= N-1:
                # We found all possible other-hand positions except the target itself.
                # Because the other hand cannot be t (collision).
                break

            # expand neighbors:
            # up to 4 moves

            # 1) L -> L+1 if d>1
            if d > 1:
                Lp = (L+1) % N
                dp_ = d-1
                # must not be 0
                if dp_ != 0:
                    v = Lp*(N-1) + (dp_-1)
                    if dist[v] < 0:
                        dist[v] = cost_u + 1
                        queue.append(v)

            # 2) L -> L-1 if d < N-1
            if d < N-1:
                Lp = (L-1) % N
                dp_ = d+1
                if dp_ != 0 and dp_ < N:
                    v = Lp*(N-1) + (dp_-1)
                    if dist[v] < 0:
                        dist[v] = cost_u + 1
                        queue.append(v)

            # 3) R -> R+1 if d < N-1
            if d < N-1:
                dp_ = d+1
                if dp_ != 0 and dp_ < N:
                    # L stays the same
                    v = L*(N-1) + (dp_-1)
                    if dist[v] < 0:
                        dist[v] = cost_u + 1
                        queue.append(v)

            # 4) R -> R-1 if d > 1
            if d > 1:
                dp_ = d-1
                if dp_ != 0:
                    # L stays the same
                    v = L*(N-1) + (dp_-1)
                    if dist[v] < 0:
                        dist[v] = cost_u + 1
                        queue.append(v)

        # now we've set newDist for dp[i]
        dp[i] = newDist

    # after Q instructions, answer is min(dp[Q])
    ans = min(dp[Q])
    print(ans)