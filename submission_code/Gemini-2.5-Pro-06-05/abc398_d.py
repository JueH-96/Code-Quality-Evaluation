import sys

def main():
    """
    Solves the Smoke problem by tracking cumulative displacements and smoke origins.
    """
    try:
        # Read input from stdin
        line1 = sys.stdin.readline().strip()
        if not line1: return
        N, R, C = map(int, line1.split())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle cases with malformed or empty input
        return

    # Map wind directions to displacement vectors (dr, dc)
    d_map = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1)
    }

    # cr, cc represents D_t, the cumulative displacement vector at time t
    cr, cc = 0, 0
    
    # active_origins stores the set of cumulative displacement vectors D_k 
    # for all times k where new smoke was generated.
    # Initially, smoke_0 is generated, corresponding to D_0 = (0,0).
    active_origins = {(0, 0)}

    # result will store the '0' or '1' for each time step
    result = []
    
    # Iterate through the wind directions for t = 1, 2, ..., N
    for move_char in S:
        # Update cumulative displacement to get D_t
        dr, dc = d_map[move_char]
        cr += dr
        cc += dc
        current_D = (cr, cc)

        # New smoke is generated at time t iff (0,0) is empty after the wind.
        # This happens iff the current cumulative displacement D_t is not an
        # origin point of any previously generated smoke cloud.
        # If it's a new origin point, we add it to our set. The set `active_origins`
        # will then represent the set of origins for the current time t.
        if current_D not in active_origins:
            active_origins.add(current_D)
        
        # Check for smoke at (R, C) at time t+0.5.
        # A smoke cloud smoke_k (where D_k is an active origin) is at position D_t - D_k.
        # We check if (R,C) = D_t - D_k for any active origin D_k.
        # This is equivalent to checking if D_k = D_t - (R,C).
        # So, we check if the target origin D_t - (R,C) exists in the set of active origins.
        target_D = (cr - R, cc - C)
        
        if target_D in active_origins:
            result.append('1')
        else:
            result.append('0')

    # Print the final result string
    print("".join(result))

if __name__ == "__main__":
    main()