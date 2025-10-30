# Read input
N, Q = map(int, input().split())
instructions = [input().split() for _ in range(Q)]

current_L = 1
current_R = 2
total_steps = 0

for H, T in instructions:
    T = int(T)
    if H == 'L':
        S = current_L
        O = current_R
    else:
        S = current_R
        O = current_L

    # Convert to 0-based indices
    S0 = S - 1
    T0 = T - 1
    N0 = N
    O0 = O - 1

    # Calculate clockwise and counter-clockwise distances
    cw_dist = (T0 - S0) % N0
    ccw_dist = (S0 - T0) % N0

    steps = 0

    if cw_dist < ccw_dist:
        # Check clockwise path
        blocked = False
        for step in range(1, cw_dist):
            current = (S0 + step) % N0
            if current == O0:
                blocked = True
                break
        if not blocked:
            steps = cw_dist
        else:
            steps = ccw_dist
    elif ccw_dist < cw_dist:
        # Check counter-clockwise path
        blocked = False
        for step in range(1, ccw_dist):
            current = (S0 - step) % N0
            if current == O0:
                blocked = True
                break
        if not blocked:
            steps = ccw_dist
        else:
            steps = cw_dist
    else:
        # Both distances are equal; check clockwise first
        blocked_cw = False
        for step in range(1, cw_dist):
            current = (S0 + step) % N0
            if current == O0:
                blocked_cw = True
                break
        if not blocked_cw:
            steps = cw_dist
        else:
            # Check counter-clockwise
            blocked_ccw = False
            for step in range(1, ccw_dist):
                current = (S0 - step) % N0
                if current == O0:
                    blocked_ccw = True
                    break
            if not blocked_ccw:
                steps = ccw_dist
            else:
                # Both paths blocked, but problem ensures it's solvable
                steps = cw_dist  # choose either

    total_steps += steps

    # Update the positions
    if H == 'L':
        current_L = T
    else:
        current_R = T

print(total_steps)