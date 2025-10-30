import sys

def solve():
    # Read input
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    R = int(line1[1])
    C = int(line1[2])
    S = sys.stdin.readline().strip()

    # Current displacement from origin (0,0) at t=0
    # This variable will store (dr[t], dc[t]) after processing wind S[t-1] at time t.
    # (dr[t], dc[t]) is the total displacement from time 0 up to time t-1 (inclusive).
    dr = 0
    dc = 0

    # Set of (dr[k], dc[k]) for k in T_gen(t)
    # T_gen(t) is the set of times <= t when smoke originated at (0,0)
    # A smoke particle generated at (0,0) at time k will have its position
    # at time t+0.5 determined by the wind movement from time k to t-1.
    # Its position at time t+0.5 will be (0 + dr[t] - dr[k], 0 + dc[t] - dc[k]).
    # We are checking if (R, C) can be expressed in this form for some k in T_gen(t).
    # (R, C) = (dr[t] - dr[k], dc[t] - dc[k])
    # This is equivalent to checking if (dr[k], dc[k]) = (dr[t] - R, dc[t] - C)
    # for some (dr[k], dc[k]) in the set {(dr[j], dc[j]) | j in T_gen(t)}.
    # The set `origin_positions` stores {(dr[j], dc[j]) | j in T_gen(t)}.
    origin_positions = set()
    origin_positions.add((0, 0)) # Represents the origin of smoke created at t=0. (dr[0], dc[0]) = (0,0).

    # Map wind character to displacement vector (delta_r, delta_c)
    # Assuming standard grid coordinates where positive r is down, positive c is right.
    move = {
        'N': (-1, 0), # Up
        'S': (1, 0),  # Down
        'W': (0, -1), # Left
        'E': (0, 1),  # Right
    }

    # Store the binary output string
    output = []

    # Simulate step by step for t=1 to N (corresponding to wind S[i] for i=0 to N-1)
    for i in range(N):
        # Wind S[i] blows at time t = i + 1
        wind = S[i]
        mdr, mdc = move[wind]

        # Update total displacement from origin at t=0 up to time t=i+1 (after wind S[i])
        # The variable (dr, dc) now holds (dr[i+1], dc[i+1]).
        dr += mdr
        dc += mdc

        # Check if (0,0) is empty after wind at time t = i+1
        # (0,0) is occupied after wind at time i+1 if any smoke particle ends up at (0,0).
        # A particle that originated at time k <= i (i.e., k in T_gen(i))
        # is at position (dr[i+1] - dr[k], dc[i+1] - dc[k]) after wind S[i].
        # It ends up at (0,0) if (dr[i+1] - dr[k], dc[i+1] - dc[k]) == (0,0),
        # which means (dr[k], dc[k]) == (dr[i+1], dc[i+1]).
        # So, (0,0) is occupied if (dr[i+1], dc[i+1]) is in {(dr[k], dc[k]) | k in T_gen(i)}.
        # The set {(dr[k], dc[k]) | k in T_gen(i)} is stored in `origin_positions` *before* checking for regeneration at time i+1.
        current_displacement = (dr, dc) # This is (dr[i+1], dc[i+1])

        if current_displacement not in origin_positions:
            # If (dr[i+1], dc[i+1]) is NOT in the set of prior origin positions,
            # it means no existing smoke particle landed on (0,0) after the wind.
            # Thus, (0,0) was empty, and new smoke is generated at (0,0) at time i+1.
            # This new smoke is a new origin. Its position relative to (0,0) at time i+1
            # is the total displacement from time 0 to i, which is (dr, dc) = (dr[i+1], dc[i+1]).
            origin_positions.add(current_displacement)
            # origin_positions now contains {(dr[k], dc[k]) | k in T_gen(i+1)}.

        # Check if smoke exists at (R, C) at time (i+1) + 0.5
        # Smoke exists at (R, C) if (R, C) is the position of any smoke particle
        # at time (i+1)+0.5.
        # A smoke particle originating at time k in T_gen(i+1)
        # is at (dr[i+1] - dr[k], dc[i+1] - dc[k]) at time (i+1)+0.5.
        # We check if (R, C) == (dr[i+1] - dr[k], dc[i+1] - dc[k]) for some k in T_gen(i+1).
        # This is equivalent to checking if (dr[k], dc[k]) == (dr[i+1] - R, dc[i+1] - C)
        # for some (dr[k], dc[k]) in the *current* `origin_positions` set (which represents T_gen(i+1)).
        target_origin_r = dr - R
        target_origin_c = dc - C
        target_origin_position = (target_origin_r, target_origin_c)

        if target_origin_position in origin_positions:
            output.append('1')
        else:
            output.append('0')

    # Print the final output string
    print("".join(output))

solve()