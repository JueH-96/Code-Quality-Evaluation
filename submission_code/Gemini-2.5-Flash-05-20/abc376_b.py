# YOUR CODE HERE
import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())

    # Initial positions (1-indexed)
    pos_L = 1
    pos_R = 2
    total_operations = 0

    for _ in range(Q):
        H, T_str = sys.stdin.readline().split()
        T = int(T_str)

        current_pos_0idx = 0  # 0-indexed position of the hand to be moved
        fixed_pos_0idx = 0    # 0-indexed position of the hand that stays fixed
        target_pos_0idx = T - 1 # 0-indexed target position for the moved hand

        # Determine which hand is moving and which is fixed
        if H == 'L':
            current_pos_0idx = pos_L - 1
            fixed_pos_0idx = pos_R - 1
        else: # H == 'R'
            current_pos_0idx = pos_R - 1
            fixed_pos_0idx = pos_L - 1
        
        # If the hand is already at the target, no operations needed
        if current_pos_0idx == target_pos_0idx:
            cost_for_instruction = 0
        else:
            # Calculate clockwise distance from current to target
            dist_cw = (target_pos_0idx - current_pos_0idx + N) % N
            # Calculate counter-clockwise distance from current to target
            dist_ccw = (current_pos_0idx - target_pos_0idx + N) % N

            cw_blocked = False
            ccw_blocked = False

            # Check if the fixed hand blocks the clockwise path
            # The fixed hand blocks if it's strictly between current_pos and target_pos on the clockwise path.
            # dist_curr_to_fixed_cw is the clockwise distance from current_pos to fixed_pos.
            dist_curr_to_fixed_cw = (fixed_pos_0idx - current_pos_0idx + N) % N
            if dist_curr_to_fixed_cw > 0 and dist_curr_to_fixed_cw < dist_cw:
                cw_blocked = True

            # Check if the fixed hand blocks the counter-clockwise path
            # The fixed hand blocks if it's strictly between current_pos and target_pos on the counter-clockwise path.
            # dist_curr_to_fixed_ccw is the counter-clockwise distance from current_pos to fixed_pos.
            dist_curr_to_fixed_ccw = (current_pos_0idx - fixed_pos_0idx + N) % N
            if dist_curr_to_fixed_ccw > 0 and dist_curr_to_fixed_ccw < dist_ccw:
                ccw_blocked = True
            
            # Determine the cost based on which paths are blocked
            if cw_blocked:
                cost_for_instruction = dist_ccw
            elif ccw_blocked:
                cost_for_instruction = dist_cw
            else:
                # Neither path is blocked, choose the shorter one
                cost_for_instruction = min(dist_cw, dist_ccw)
        
        total_operations += cost_for_instruction

        # Update the position of the moved hand (store as 1-indexed)
        if H == 'L':
            pos_L = T
        else:
            pos_R = T

    print(total_operations)

# Call the solve function to run the program
solve()