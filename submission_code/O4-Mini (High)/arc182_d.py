import sys
import threading
def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); M = int(line[1])
    A = list(map(int, data.readline().split()))
    B = list(map(int, data.readline().split()))
    # Special case M == 2
    if M == 2:
        # Only good sequences are alternating 0/1. You cannot change any element without colliding.
        if A == B:
            print(0)
        else:
            print(-1)
        return
    # Precompute per-index costs and directions
    # cost0[i], cost1[i], dir0[i], dir1[i]
    N0 = N
    cost0 = [0]*N0
    cost1 = [0]*N0
    dir0  = [0]*N0
    dir1  = [0]*N0
    for i in range(N0):
        a = A[i]; b = B[i]
        # forward delta
        d = b - a
        # mod M to [0,M)
        if d < 0 or d >= M:
            d %= M
        # forward steps = d, backward = M-d
        fwd = d
        bwd = M - d
        # minimal cost
        if fwd <= bwd:
            c0 = fwd; s0 = 1
            c1 = bwd; s1 = -1
        else:
            c0 = bwd; s0 = -1
            c1 = fwd; s1 = 1
        cost0[i] = c0
        cost1[i] = c1
        dir0[i]  = s0
        dir1[i]  = s1
    # Precompute lim and rit for adjacent pairs
    # lim[i] = (A[i] - A[i+1]) mod M
    # rit[i] = (B[i] - B[i+1]) mod M
    N1 = N0 - 1
    lim = [0]*N1
    rit = [0]*N1
    for i in range(N1):
        x = A[i] - A[i+1]
        if x < 0 or x >= M:
            x %= M
        lim[i] = x
        y = B[i] - B[i+1]
        if y < 0 or y >= M:
            y %= M
        rit[i] = y
    # DP on chain
    INF = 10**30
    # dp_prev[0] = min cost up to i at d_i=0; dp_prev[1] = ... d_i=1
    dp0 = cost0[0]
    dp1 = cost1[0]
    # iterate edges
    for i in range(N1):
        ndp0 = INF
        ndp1 = INF
        # Next index is i+1
        # Preload next costs and dirs
        c_next0 = cost0[i+1]; c_next1 = cost1[i+1]
        dir_next0 = dir0[i+1]; dir_next1 = dir1[i+1]
        # Current lim/rit
        limv = lim[i]; ritv = rit[i]
        Ai = A[i]; Ai1 = A[i+1]
        # Try d_next = 0 or 1
        # d_next = 0
        c2 = c_next0
        for d_prev in (0,1):
            # previous dp value
            dp_prev_val = dp0 if d_prev == 0 else dp1
            if dp_prev_val >= INF:
                continue
            # previous cost for checking zero-move
            c1 = cost0[i] if d_prev == 0 else cost1[i]
            # dirs
            dir_prev = dir0[i] if d_prev == 0 else dir1[i]
            dir_cur  = dir_next0
            allowed = False
            # both zero
            if c1 == 0 and c2 == 0:
                allowed = True
            # single-move case
            elif c1 == 0 or c2 == 0:
                if c2 == 0:
                    # moving = i
                    neighbor = Ai1
                    A_move = Ai
                    dir_move = dir_prev
                    cost_move = c1
                else:
                    # c1 == 0, moving = i+1
                    neighbor = Ai
                    A_move = Ai1
                    dir_move = dir_cur
                    cost_move = c2
                # compute diff
                if dir_move == 1:
                    diff = neighbor - A_move
                    if diff < 0 or diff >= M:
                        diff %= M
                else:
                    diff = A_move - neighbor
                    if diff < 0 or diff >= M:
                        diff %= M
                # forbidden if neighbor lies in (A_move to B_move] along dir_move
                if not (1 <= diff <= cost_move):
                    allowed = True
            else:
                # both moving
                if dir_prev == -dir_cur:
                    # all moves same sign -> need check pass-through 0
                    if dir_prev == 1:
                        # plus direction: forbidden if limv >= ritv
                        if limv < ritv:
                            allowed = True
                    else:
                        # minus direction: forbidden if limv <= ritv
                        if limv > ritv:
                            allowed = True
                else:
                    # mixed moves: always possible for M>=3
                    allowed = True
            if allowed:
                val = dp_prev_val + c2
                if val < ndp0:
                    ndp0 = val
        # d_next = 1
        c2 = c_next1
        for d_prev in (0,1):
            dp_prev_val = dp0 if d_prev == 0 else dp1
            if dp_prev_val >= INF:
                continue
            c1 = cost0[i] if d_prev == 0 else cost1[i]
            dir_prev = dir0[i] if d_prev == 0 else dir1[i]
            dir_cur  = dir_next1
            allowed = False
            if c1 == 0 and c2 == 0:
                allowed = True
            elif c1 == 0 or c2 == 0:
                if c2 == 0:
                    neighbor = Ai1
                    A_move = Ai
                    dir_move = dir_prev
                    cost_move = c1
                else:
                    neighbor = Ai
                    A_move = Ai1
                    dir_move = dir_cur
                    cost_move = c2
                if dir_move == 1:
                    diff = neighbor - A_move
                    if diff < 0 or diff >= M:
                        diff %= M
                else:
                    diff = A_move - neighbor
                    if diff < 0 or diff >= M:
                        diff %= M
                if not (1 <= diff <= cost_move):
                    allowed = True
            else:
                if dir_prev == -dir_cur:
                    if dir_prev == 1:
                        if limv < ritv:
                            allowed = True
                    else:
                        if limv > ritv:
                            allowed = True
                else:
                    allowed = True
            if allowed:
                val = dp_prev_val + c2
                if val < ndp1:
                    ndp1 = val
        dp0, dp1 = ndp0, ndp1
        # early stop if both INF
        if dp0 >= INF and dp1 >= INF:
            print(-1)
            return
    res = dp0 if dp0 < dp1 else dp1
    if res >= INF:
        print(-1)
    else:
        print(res)

if __name__ == "__main__":
    main()