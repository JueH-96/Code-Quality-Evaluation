# YOUR CODE HERE
import sys

def dist_cw_1indexed(u, v, N):
    """Clockwise distance from u to v on a ring of size N."""
    # Parts are 1 to N.
    # The displacement v - u can be negative. Adding N makes it non-negative
    # before taking modulo N, effectively wrapping around the ring.
    return (v - u + N) % N

def dist_ccw_1indexed(u, v, N):
    """Counter-clockwise distance from u to v on a ring of size N."""
    # Parts are 1 to N.
    # The displacement u - v can be negative. Adding N makes it non-negative
    # before taking modulo N, effectively wrapping around the ring.
    return (u - v + N) % N

def solve():
    # Read N and Q from the first line
    N, Q = map(int, sys.stdin.readline().split())
    
    # Initial positions of the left and right hands
    l_pos = 1
    r_pos = 2
    
    # Initialize total operations counter
    total_ops = 0
    
    # Process each instruction
    for _ in range(Q):
        # Read instruction: hand (H) and target part (T)
        H, T = sys.stdin.readline().split()
        T = int(T)
        
        # Determine the moving hand (src), the obstacle hand (obs), and the target part (tgt)
        if H == 'L':
            src = l_pos
            obs = r_pos
            tgt = T
        else: # H == 'R'
            src = r_pos
            obs = l_pos
            tgt = T

        # If the moving hand is already at the target, 0 operations are needed.
        if src == tgt:
            ops = 0
        else:
            # Calculate the clockwise and counter-clockwise distances from source to target
            d_cw = dist_cw_1indexed(src, tgt, N)
            d_ccw = dist_ccw_1indexed(src, tgt, N)
            
            # Calculate the clockwise and counter-clockwise distances from source to obstacle
            # The problem guarantees that the two hands are always on different parts,
            # so src != obs. This means d_obs_cw > 0 and d_obs_ccw > 0.
            d_obs_cw = dist_cw_1indexed(src, obs, N)
            d_obs_ccw = dist_ccw_1indexed(src, obs, N)

            # A directed path (CW or CCW) from src to tgt is blocked
            # if the obstacle is strictly between src and tgt on that path.
            # This means the obstacle is reached in < d steps
            # in that direction from src, where d is the distance to tgt.
            # Since src != obs and src != tgt are guaranteed (for src != tgt case),
            # checking d_obs_cw < d_cw is sufficient.
            is_cw_blocked = (d_obs_cw < d_cw)
            is_ccw_blocked = (d_obs_ccw < d_ccw)
            
            # The problem guarantees achievable instructions.
            # Under the ring topology and the movement constraints, this implies
            # that at most one of the two directed paths (CW or CCW) from src to tgt
            # is blocked by the obstacle.
            if is_cw_blocked:
                # If the clockwise path is blocked, must take the counter-clockwise path.
                ops = d_ccw
            elif is_ccw_blocked:
                # If the counter-clockwise path is blocked, must take the clockwise path.
                ops = d_cw
            else:
                # If neither path is blocked, take the shorter one.
                ops = min(d_cw, d_ccw)

        # Add the operations for this instruction to the total
        total_ops += ops
        
        # Update the position of the hand that moved
        if H == 'L':
            l_pos = tgt
        else:
            r_pos = tgt

    # Print the total number of operations
    print(total_ops)

# Execute the solve function when the script is run
solve()