# YOUR CODE HERE
def main():
    import sys,sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    # Read instructions as list of (hand, target)
    instr = []
    for i in range(Q):
        h = next(it)
        t = int(next(it))
        instr.append((h, t))
    INF = 10**12

    # d(a,b): clockwise distance from a to b in 1-indexed ring.
    def d(a, b):
        diff = b - a
        if diff < 0:
            diff += N
        return diff

    # free_cost(p,q,t): minimal cost (number of one‐step moves) to move from p to t 
    # following the circular path that does NOT contain q.
    # (Assumes t != q.)
    def free_cost(p, q, t):
        dval = d(p, t)
        diff = d(p, q)
        if diff < dval:
            return N - dval
        else:
            return dval

    def next_cw(x):
        # Next clockwise part
        return x+1 if x < N else 1

    def prev_val(x):
        # Previous clockwise part
        return x-1 if x > 1 else N

    # process_move_L: when processing an instruction "L target"
    # we are moving the left hand (currently at position p) while the 
    # right hand is at obs. (base is our current cumulative cost.)
    # Returns list of outcomes (new_obs, total_cost), where the new state becomes (target, new_obs)
    def process_move_L(p, obs, target, base):
        outs = []
        if p == target:
            outs.append((obs, base))
        else:
            if target != obs:
                cost_move = free_cost(p, obs, target)
                outs.append((obs, base + cost_move))
            else:
                # The destination target equals the current obstacle: we must reposition.
                cw = next_cw(obs)
                if cw != p:
                    cost1 = free_cost(obs, p, cw)  # move obstacle from obs to cw (with p fixed)
                    cost2 = free_cost(p, cw, target)  # then move left hand to target
                    outs.append((cw, base + cost1 + cost2))
                ccw = prev_val(obs)
                if ccw != p:
                    cost1 = free_cost(obs, p, ccw)
                    cost2 = free_cost(p, ccw, target)
                    outs.append((ccw, base + cost1 + cost2))
        return outs

    # process_move_R: for an instruction "R target" we are moving the right hand.
    # Here the moving hand is at p and the obstacle (left hand) is at obs.
    # New outcomes are (new_obs,total_cost) meaning the new state becomes (new_obs, target).
    def process_move_R(p, obs, target, base):
        outs = []
        if p == target:
            outs.append((obs, base))
        else:
            if target != obs:
                cost_move = free_cost(p, obs, target)
                outs.append((obs, base + cost_move))
            else:
                cw = next_cw(obs)
                if cw != p:
                    cost1 = free_cost(obs, p, cw)
                    cost2 = free_cost(p, cw, target)
                    outs.append((cw, base + cost1 + cost2))
                ccw = prev_val(obs)
                if ccw != p:
                    cost1 = free_cost(obs, p, ccw)
                    cost2 = free_cost(p, ccw, target)
                    outs.append((ccw, base + cost1 + cost2))
        return outs

    # Our DP will “compress” the state.
    # When a left–hand move is executed, the resulting state is of type L:
    # state = (currL, r) with left hand fixed (currL == target) and free parameter r = right hand.
    # Similarly, a right–hand move gives a state of type R: (l, currR) where currR == target.
    #
    # We represent dp_L as a list (indices 1..N) where dp_L[r] is the minimal cost 
    # attaining state (currL, r). (currL is stored separately.)
    # And dp_R as a list (indices 1..N) where dp_R[l] is the minimal cost for (l, currR).
    #
    # Initially, we are in a neutral state: (1,2) at cost 0.
    dp_L = None  # for type L states
    currL = None
    dp_R = None  # for type R states
    currR = None

    neut_L = 1
    neut_R = 2
    neut_cost = 0

    # Process instructions one by one.
    for (hand, target) in instr:
        if hand == "L":
            # When processing an "L" instruction the left hand is being moved,
            # and the new state is of type L: (target, X)
            new_dp = [INF]*(N+1)  # indices 1..N for the right-hand free coordinate.
            if dp_L is None and dp_R is None:
                # Use neutral state (1,2)
                for (new_obs, cost_here) in process_move_L(neut_L, neut_R, target, neut_cost):
                    if cost_here < new_dp[new_obs]:
                        new_dp[new_obs] = cost_here
            else:
                if dp_L is not None:
                    # Previous state from type L: (currL, r)
                    p_fixed = currL
                    for r in range(1, N+1):
                        curc = dp_L[r]
                        if curc < INF:
                            for (new_obs, add_cost) in process_move_L(p_fixed, r, target, curc):
                                if add_cost < new_dp[new_obs]:
                                    new_dp[new_obs] = add_cost
                if dp_R is not None:
                    # Also use state from type R: (l, currR)
                    fixed_obs = currR
                    for l in range(1, N+1):
                        curc = dp_R[l]
                        if curc < INF:
                            for (new_obs, add_cost) in process_move_L(l, fixed_obs, target, curc):
                                if add_cost < new_dp[new_obs]:
                                    new_dp[new_obs] = add_cost
            dp_L = new_dp
            currL = target
            dp_R = None
            currR = None
        else:
            # Processing an "R" instruction.
            # The new state has type R: (X, target) where X is the free left-hand coordinate.
            new_dp = [INF]*(N+1)  # indices 1..N for the free coordinate.
            if dp_L is None and dp_R is None:
                # Use neutral state (1,2) but now move right hand from 2 (with obstacle 1)
                for (new_obs, cost_here) in process_move_R(neut_R, neut_L, target, neut_cost):
                    if cost_here < new_dp[new_obs]:
                        new_dp[new_obs] = cost_here
            else:
                if dp_R is not None:
                    # From type R state: (l, currR)
                    p_fixed = currR
                    for l in range(1, N+1):
                        curc = dp_R[l]
                        if curc < INF:
                            for (new_obs, add_cost) in process_move_R(p_fixed, l, target, curc):
                                if add_cost < new_dp[new_obs]:
                                    new_dp[new_obs] = add_cost
                if dp_L is not None:
                    # Also from type L state: (currL, r)
                    fixed_obs = currL
                    for r in range(1, N+1):
                        curc = dp_L[r]
                        if curc < INF:
                            for (new_obs, add_cost) in process_move_R(r, fixed_obs, target, curc):
                                if add_cost < new_dp[new_obs]:
                                    new_dp[new_obs] = add_cost
            dp_R = new_dp
            currR = target
            dp_L = None
            currL = None

    # Our final answer is the minimum cost in the active dp.
    ans = INF
    if dp_L is not None:
        for val in dp_L:
            if val < ans:
                ans = val
    if dp_R is not None:
        for val in dp_R:
            if val < ans:
                ans = val
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()