# YOUR CODE HERE
import sys

def solve():
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    R = int(line1[1])
    C = int(line1[2])
    S = sys.stdin.readline().strip()

    # Displacement map for each wind direction
    # (dr, dc) represents (delta_row, delta_col)
    # According to problem description:
    # 'N': (r, c) -> (r-1, c) => dr = -1, dc = 0
    # 'W': (r, c) -> (r, c-1) => dr = 0, dc = -1
    # 'S': (r, c) -> (r+1, c) => dr = 1, dc = 0
    # 'E': (r, c) -> (r, c+1) => dr = 0, dc = 1
    displacement = {
        'N': (-1, 0),
        'W': (0, -1),
        'S': (1, 0),
        'E': (0, 1),
    }

    # Cumulative displacement from time 0 for a particle starting at (0,0)
    # (cum_dr, cum_dc) at step t is CumD_t = (CumDR_t, CumDC_t)
    # Initialize for t=0
    cum_dr = 0
    cum_dc = 0

    # Set of (CumDR_i, CumDC_i) for all active source times i.
    # An active source time i means a particle originated at (0,0) at time i
    # (i=0 for initial, i>0 for regeneration) and this source is distinct
    # from other active sources up to that point.
    # The set active_cum_d at the beginning of iteration t contains (CumDR_i, CumDC_i)
    # for all active source times i <= t-1.
    # Initially, only the particle starting at t=0 is active. CumD_0 = (0,0).
    active_cum_d = set()
    active_cum_d.add((0, 0)) # Represents CumD_0 at time i=0

    result = []

    # Loop through time steps t from 1 to N.
    # Wind S[t-1] blows at time t. We check smoke at t+0.5.
    for t in range(1, N + 1):
        # Get displacement for wind at time t (S[t-1])
        dr, dc = displacement[S[t-1]]

        # Update cumulative displacement after wind at time t.
        # This new (cum_dr, cum_dc) is CumD_t.
        cum_dr += dr
        cum_dc += dc

        # Smoke is at (R, C) at time t+0.5 iff (R, C) = CumD_t - CumD_i
        # for some active source time i <= t-1.
        # This is equivalent to CumD_i = CumD_t - (R, C).
        # We need to check if (CumD_t - R, CumDC_t - C) is in the set of
        # active cumulative displacements from times 0 to t-1.
        target_cum_dr = cum_dr - R
        target_cum_dc = cum_dc - C

        # Check if the required CumD_i exists in the set active_cum_d (which holds CumD_j for j in ActiveSet_{t-1})
        if (target_cum_dr, target_cum_dc) in active_cum_d:
            result.append('1')
        else:
            result.append('0')

        # Determine if (0,0) is empty at time t+0.5.
        # (0,0) is empty at t+0.5 iff (0,0) is NOT reached by any active source
        # particle after wind at time t.
        # (0,0) is reached by an active source i iff (0,0) = CumD_t - CumD_i for some i in ActiveSet_{t-1}.
        # This is equivalent to CumD_t = CumD_i for some i in ActiveSet_{t-1}.
        # So, (0,0) is empty at t+0.5 iff (cum_dr, cum_dc) is NOT in active_cum_d.
        is_origin_hit = (cum_dr, cum_dc) in active_cum_d

        # If (0,0) was empty at time t+0.5, a new particle is generated at (0,0).
        # This particle effectively starts as a new active source at time t.
        # We add CumD_t to the set active_cum_d. This set will now contain
        # CumD_i for i in ActiveSet_t for the next iteration (t+1).
        if not is_origin_hit:
             active_cum_d.add((cum_dr, cum_dc)) # Represents CumD_t for active source originated at time t

    print("".join(result))

solve()