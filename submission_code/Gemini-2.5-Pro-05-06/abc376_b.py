import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    
    # Using 0-indexed positions internally for parts 0 to N-1
    # Initial positions: L on part 1 (0-indexed: 0), R on part 2 (0-indexed: 1)
    l_pos = 0 
    r_pos = 1
    
    total_ops = 0
    
    for _ in range(Q):
        line = sys.stdin.readline().split()
        hand_char = line[0]
        target_1_idx = int(line[1])
        
        # Convert target to 0-indexed
        target_0_idx = target_1_idx - 1
        
        curr_hand_pos = -1
        other_hand_pos = -1
        
        if hand_char == 'L':
            curr_hand_pos = l_pos
            other_hand_pos = r_pos
        else: # hand_char == 'R'
            curr_hand_pos = r_pos
            other_hand_pos = l_pos
            
        if curr_hand_pos == target_0_idx:
            # No operations needed if already at target
            ops_this_instruction = 0
        else:
            # Calculate clockwise distance from curr_hand_pos to target_0_idx
            dist_cw = (target_0_idx - curr_hand_pos + N) % N
            
            # Calculate counter-clockwise distance from curr_hand_pos to target_0_idx
            dist_ccw = (curr_hand_pos - target_0_idx + N) % N
            
            path_cw_blocked = False
            # Clockwise distance from curr_hand_pos to other_hand_pos
            dist_curr_to_other_cw = (other_hand_pos - curr_hand_pos + N) % N
            # If other_hand_pos is an intermediate node on the CW path from curr_hand_pos to target_0_idx,
            # then the CW distance from curr_hand_pos to other_hand_pos must be less than dist_cw.
            # Constraints guarantee: other_hand_pos != curr_hand_pos and other_hand_pos != target_0_idx.
            # Also, curr_hand_pos != target_0_idx in this 'else' block, so dist_cw > 0.
            if dist_curr_to_other_cw < dist_cw:
                path_cw_blocked = True
            
            path_ccw_blocked = False
            # Counter-clockwise distance from curr_hand_pos to other_hand_pos
            dist_curr_to_other_ccw = (curr_hand_pos - other_hand_pos + N) % N
            # If other_hand_pos is an intermediate node on the CCW path from curr_hand_pos to target_0_idx,
            # then the CCW distance from curr_hand_pos to other_hand_pos must be less than dist_ccw.
            if dist_curr_to_other_ccw < dist_ccw:
                path_ccw_blocked = True

            if not path_cw_blocked and not path_ccw_blocked:
                ops_this_instruction = min(dist_cw, dist_ccw)
            elif not path_cw_blocked:
                ops_this_instruction = dist_cw
            elif not path_ccw_blocked:
                ops_this_instruction = dist_ccw
            else:
                # This case should not be reached due to problem guarantees.
                # It implies other_hand_pos is strictly between curr_hand_pos and target_0_idx
                # in BOTH clockwise and counter-clockwise directions, which is impossible.
                # assert False, "Error: Both paths determined to be blocked." 
                pass


        total_ops += ops_this_instruction
        
        # Update hand position
        if hand_char == 'L':
            l_pos = target_0_idx
        else: # hand_char == 'R'
            r_pos = target_0_idx
            
    print(total_ops)

if __name__ == '__main__':
    main()