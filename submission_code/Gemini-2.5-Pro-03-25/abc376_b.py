# YOUR CODE HERE
import sys

# Function to calculate clockwise distance from a to b on a ring of size N
# Parts are numbered 1 to N
def cw_dist(a, b, N):
    """Calculates the number of steps to move clockwise from part a to part b."""
    # If a == b, distance is 0 steps.
    # Otherwise, the number of steps is (b - a + N) % N.
    # The modulo operator % in Python handles negative inputs correctly 
    # to produce a result in [0, N-1].
    # Example N=6: cw_dist(1, 5, 6) = (5 - 1 + 6) % 6 = 4 % 6 = 4. Path 1->2->3->4->5 takes 4 steps.
    # Example N=6: cw_dist(5, 1, 6) = (1 - 5 + 6) % 6 = 2 % 6 = 2. Path 5->6->1 takes 2 steps.
    return (b - a + N) % N

# Function to calculate counter-clockwise distance from a to b
def ccw_dist(a, b, N):
    """Calculates the number of steps to move counter-clockwise from part a to part b."""
    # If a == b, distance is 0 steps.
    # Otherwise, the number of steps is (a - b + N) % N.
    # Example N=6: ccw_dist(1, 5, 6) = (1 - 5 + 6) % 6 = 2 % 6 = 2. Path 1->6->5 takes 2 steps.
    # Example N=6: ccw_dist(5, 1, 6) = (5 - 1 + 6) % 6 = 10 % 6 = 4. Path 5->4->3->2->1 takes 4 steps.
    return (a - b + N) % N

def solve():
    # Read N (number of parts) and Q (number of instructions)
    N, Q = map(int, sys.stdin.readline().split())
    
    # Initial positions of left (L) and right (R) hands
    l_pos = 1
    r_pos = 2
    
    # Variable to accumulate the total number of operations
    total_ops = 0
    
    # Process each instruction
    for _ in range(Q):
        # Read the instruction: Hand H and Target T
        line = sys.stdin.readline().split()
        H = line[0] # Hand to move ('L' or 'R')
        T = int(line[1]) # Target part number
        
        # Determine current position, target position, and fixed hand position
        if H == 'L':
            curr = l_pos
            target = T
            fixed = r_pos
        else: # H == 'R'
            curr = r_pos
            target = T
            fixed = l_pos
            
        # If the hand is already at the target position, no operations are needed
        if curr == target:
            ops_needed = 0
        else:
            # Calculate the lengths of the clockwise and counter-clockwise paths
            d_cw = cw_dist(curr, target, N) # Clockwise path length
            d_ccw = ccw_dist(curr, target, N) # Counter-clockwise path length
            
            # Determine if the fixed hand's position blocks the clockwise path.
            # The fixed hand blocks the clockwise path if it lies on the path segment
            # strictly between the current position and the target position, moving clockwise.
            
            # Calculate the clockwise distance from the current position to the fixed hand's position
            dist_curr_fixed_cw = cw_dist(curr, fixed, N)
            
            # The fixed hand is on the clockwise path segment if its clockwise distance from `curr`
            # is less than the clockwise distance from `curr` to `target`.
            # Note: We know `curr != target` and `curr != fixed`. The problem guarantees `target != fixed`.
            # Thus, `d_cw > 0` and `dist_curr_fixed_cw > 0`.
            is_fixed_on_cw_path = (dist_curr_fixed_cw < d_cw)

            if is_fixed_on_cw_path:
                # If the fixed point blocks the clockwise path, we must use the counter-clockwise path.
                ops_needed = d_ccw
            else:
                # If the fixed point does not block the clockwise path (it must block the counter-clockwise path),
                # we must use the clockwise path.
                ops_needed = d_cw

        # Add the operations required for this instruction to the total count
        total_ops += ops_needed
        
        # Update the position of the moved hand
        if H == 'L':
            l_pos = target
        else: # H == 'R'
            r_pos = target

    # Print the final total minimum number of operations
    print(total_ops)

# Execute the solve function to run the program logic
solve()