import sys

# Read input
line1 = sys.stdin.readline().split()
N = int(line1[0])
R = int(line1[1])
C = int(line1[2])
S = sys.stdin.readline().strip()

# current_r, current_c track the cumulative displacement
# from the initial campfire location (0,0) according to wind.
# (r_t, c_t) is the cumulative displacement at time t.
current_r = 0
current_c = 0

# V stores the set of cumulative displacements (r_k, c_k)
# for all times k (0 <= k < current_time) where smoke was generated at (0,0).
# A particle generated at time k is at position (r_t - r_k, c_t - c_k)
# at time t after wind.
# The smoke initially at t=0 corresponds to k=0, displacement (r_0, c_0) = (0,0).
# V is updated during the loop for the *next* iteration.
# So, at the start of iteration t (for wind S[t-1]), V contains (r_k, c_k) for k < t
# where smoke was generated.
V = set()
V.add((0, 0)) # Smoke generated at time 0 corresponds to (r_0, c_0) = (0,0)

# Result string builder
result = []

# Simulate time steps t = 1 to N
# At each step t, wind S[t-1] blows, then new smoke might generate at (0,0).
# We check for smoke at (R,C) at time t+0.5, which is *after* the wind at time t.
for t in range(1, N + 1):
    # Get the wind direction for time t (S[t-1])
    move = S[t-1]

    # Update the cumulative displacement based on the wind up to time t
    if move == 'N':
        current_r -= 1
    elif move == 'S':
        current_r += 1
    elif move == 'W':
        current_c -= 1
    elif move == 'E':
        current_c += 1

    # (rt, ct) is the cumulative displacement at time t, relative to t=0.
    rt, ct = current_r, current_c

    # Check if smoke exists at target cell (R, C) at time t+0.5.
    # Smoke exists at (R, C) at time t+0.5 if it is the position of any
    # particle that originated at (0,0) at some time k (where smoke was
    # generated, 0 <= k < t), after applying the wind up to time t.
    # The position of such a particle is (r_t - r_k, c_t - c_k).
    # We check if (R, C) = (r_t - r_k, c_t - c_k) for some (r_k, c_k) in V.
    # This is equivalent to checking if (r_t - R, c_t - C) is in V.
    # At the start of this loop iteration (for time t), V contains
    # (r_k, c_k) for all k < t where smoke was generated.
    target_relative_start_pos = (rt - R, ct - C)

    if target_relative_start_pos in V:
        result.append('1')
    else:
        result.append('0')

    # After checking the target cell, determine if new smoke is generated
    # at (0,0) at time t.
    # New smoke is generated if (0,0) is empty *after* wind at time t.
    # (0,0) is empty after wind at time t iff no existing particle (from k < t)
    # is at (0,0) after wind.
    # An existing particle from time k < t is at (r_t - r_k, c_t - c_k).
    # (0,0) is occupied iff (r_t - r_k, c_t - c_k) == (0,0) for some (r_k, c_k) in V.
    # This is equivalent to checking if (r_t, c_t) is in V.
    is_origin_occupied_after_wind = (rt, ct) in V

    # If (0,0) was empty after wind, new smoke appears at time t.
    # This new smoke particle effectively starts its journey from (0,0) at time t.
    # Its "relative start position" corresponds to the cumulative displacement
    # at time t, which is (r_t, c_t). We add this to V for future checks (times > t).
    # V will now contain (r_k, c_k) for all k <= t where smoke was generated.
    if not is_origin_occupied_after_wind:
        V.add((rt, ct))

# Print the final result string
sys.stdout.write("".join(result) + "
")