import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())

    l_pos = 1  # Current position of the left hand
    r_pos = 2  # Current position of the right hand
    total_operations = 0

    def min_dist_on_ring(p1, p2, N_val):
        """Calculates the minimum distance between p1 and p2 on a ring of size N_val."""
        # The distance on a line
        d_linear = abs(p1 - p2)
        # The distance around the ring (the other way)
        d_ring_other_way = N_val - d_linear
        return min(d_linear, d_ring_other_way)

    for _ in range(Q):
        H, T_str = sys.stdin.readline().split()
        T = int(T_str)

        if H == 'L':
            # Instruction is for the Left hand
            if l_pos == T:
                # Left hand is already at the target, no operations needed for this instruction
                continue
            
            # Left hand (H) needs to move to T
            # Right hand (O) is the other hand, at r_pos
            
            current_H_pos = l_pos
            current_O_pos = r_pos
            
            if T == current_O_pos:
                # Case B: Target position T is currently occupied by the other hand (Right hand)
                # O (Right hand) must move one step to clear T. Cost = 1 operation.
                # Then H (Left hand) moves from current_H_pos to T.
                
                cost_H_move = min_dist_on_ring(current_H_pos, T, N)
                total_operations += (cost_H_move + 1) # +1 for O moving out of the way
                
                # Update positions: H moves to T
                l_pos = T
                
                # O (r_pos) moves 1 step. It must not move to the new l_pos (which is T).
                # Calculate potential clockwise and counter-clockwise adjacent positions for r_pos
                r_pos_cw = (current_O_pos % N) + 1  # If current_O_pos is N, (N%N)+1 = 1
                r_pos_ccw = (current_O_pos - 2 + N) % N + 1 # If current_O_pos is 1, (1-2+N)%N+1 = (N-1)%N+1 = N
                
                # Choose the adjacent spot for r_pos that is not the new l_pos (T)
                if r_pos_cw != l_pos:
                    r_pos = r_pos_cw
                else:
                    # Since N >= 3, the two adjacent spots (cw and ccw) are distinct.
                    # If cw is blocked by l_pos, ccw must be available.
                    r_pos = r_pos_ccw
                    
            else:
                # Case A: Target position T is not occupied by the other hand (Right hand)
                # H (Left hand) moves directly from current_H_pos to T. O (Right hand) stays put.
                cost_H_move = min_dist_on_ring(current_H_pos, T, N)
                total_operations += cost_H_move
                
                # Update position: H moves to T
                l_pos = T
                # r_pos remains unchanged

        else: # H == 'R'
            # Instruction is for the Right hand
            if r_pos == T:
                # Right hand is already at the target, no operations needed for this instruction
                continue
            
            # Right hand (H) needs to move to T
            # Left hand (O) is the other hand, at l_pos
            
            current_H_pos = r_pos
            current_O_pos = l_pos
            
            if T == current_O_pos:
                # Case B: Target position T is currently occupied by the other hand (Left hand)
                # O (Left hand) must move one step to clear T. Cost = 1 operation.
                # Then H (Right hand) moves from current_H_pos to T.
                
                cost_H_move = min_dist_on_ring(current_H_pos, T, N)
                total_operations += (cost_H_move + 1) # +1 for O moving out of the way
                
                # Update positions: H moves to T
                r_pos = T
                
                # O (l_pos) moves 1 step. It must not move to the new r_pos (which is T).
                l_pos_cw = (current_O_pos % N) + 1
                l_pos_ccw = (current_O_pos - 2 + N) % N + 1
                
                if l_pos_cw != r_pos:
                    l_pos = l_pos_cw
                else:
                    l_pos = l_pos_ccw
            else:
                # Case A: Target position T is not occupied by the other hand (Left hand)
                # H (Right hand) moves directly from current_H_pos to T. O (Left hand) stays put.
                cost_H_move = min_dist_on_ring(current_H_pos, T, N)
                total_operations += cost_H_move
                
                # Update position: H moves to T
                r_pos = T
                # l_pos remains unchanged
                
    sys.stdout.write(str(total_operations) + '
')

solve()