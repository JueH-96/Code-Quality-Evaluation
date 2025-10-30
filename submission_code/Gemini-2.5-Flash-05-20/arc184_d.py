import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Store points mapped by their X-coordinate.
    # points_by_x[x_val - 1] will store (y_val - 1).
    # This allows us to work with 0-indexed coordinates [0, N-1] internally,
    # where the index in points_by_x array represents the X-coordinate.
    points_by_x = [0] * N 
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points_by_x[x-1] = y-1 # Store Y-coordinate (0-indexed) at X-coordinate (0-indexed) position
    
    # Use a set to store unique frozensets representing the possible remaining balls.
    possible_remaining_sets = set()
    
    # Case 1: No operations performed. All N balls remain.
    # Balls are 0-indexed integers from 0 to N-1.
    initial_set_of_balls = frozenset(range(N))
    possible_remaining_sets.add(initial_set_of_balls)
    
    # Case 2: Perform an operation by choosing a ball k from the initial set of balls.
    # According to the problem analysis, all reachable sets are either the initial set
    # or a set obtained by picking a single ball 'k' from the initial set and applying the rule.
    
    # Iterate through each ball k_idx (0 to N-1) as the chosen ball.
    for k_idx in range(N):
        # The chosen ball k_idx itself always remains.
        current_set_of_remaining_balls_list = [k_idx] 
        
        # Get coordinates of ball k (0-indexed)
        # X-coordinate is its index, Y-coordinate is value from points_by_x
        k_x = k_idx
        k_y = points_by_x[k_idx]
        
        # Check all other balls i_idx (0 to N-1)
        for i_idx in range(N):
            if i_idx == k_idx:
                continue # Skip ball k itself, as it's already added.
            
            # Get coordinates of ball i (0-indexed)
            i_x = i_idx
            i_y = points_by_x[i_idx]
            
            # A ball i remains if it's "anti-monotonically related" to k:
            # (X_i < X_k and Y_i > Y_k) OR (X_i > X_k and Y_i < Y_k)
            # Using 0-indexed values:
            if (i_x < k_x and i_y > k_y) or \
               (i_x > k_x and i_y < k_y):
                current_set_of_remaining_balls_list.append(i_idx)
        
        # Convert the list of remaining balls to a frozenset for hashability
        # and add it to our collection of possible unique sets.
        possible_remaining_sets.add(frozenset(current_set_of_remaining_balls_list))
            
    # The answer is the total count of unique possible remaining sets.
    # The modulo 998244353 is typically for problems where answers can be very large.
    # Here, the maximum count (N+1 = 301) is small, so we print the count directly.
    print(len(possible_remaining_sets))

solve()