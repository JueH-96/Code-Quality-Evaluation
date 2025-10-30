# YOUR CODE HERE
import sys

# Function to perform the simulation and find final rows
def solve():
    # Read input dimensions H (height), W (width), N (number of bars)
    H, W, N = map(int, sys.stdin.readline().split())
    
    # Read bar definitions and store them. Keep track of initial info.
    # bars_data stores a list of dictionaries, each containing info for one bar:
    # {'id': original index (0 to N-1), 'R': initial row, 'C': initial column, 'L': length}
    bars_data = [] 
    for i in range(N):
        R, C, L = map(int, sys.stdin.readline().split())
        # Store initial data. Rows R and columns C are 1-based as per problem statement.
        bars_data.append({'id': i, 'R': R, 'C': C, 'L': L})

    # current_pos maps bar_id (0 to N-1) -> current row R (1-based)
    # Initialize with initial rows from input.
    current_pos = {} 
    for i in range(N):
        current_pos[i] = bars_data[i]['R']

    # row_map maps row number -> dictionary {bar_id: (C_start, C_end)}
    # This structure efficiently stores which bars occupy which column intervals at each row.
    # C_start and C_end are 1-based column indices defining the interval [C_start, C_end].
    row_map = {} 
    
    # Populate initial row_map based on initial bar positions
    for i in range(N):
        bar = bars_data[i]
        r, c, l = bar['R'], bar['C'], bar['L']
        
        # If this row is not yet in row_map, initialize its dictionary
        if r not in row_map:
            row_map[r] = {}
        
        # Store bar i's ID and its column interval [C, C+L-1] at its current row r
        row_map[r][i] = (c, c + l - 1) 

    # Main simulation loop: continues as long as any bar moves during a full pass (i=0..N-1)
    active = True # Flag to track if any bar moved in the current pass
    while active:
        active = False # Reset flag at the beginning of each pass
        
        # Process bars sequentially in order of their original index i = 0 to N-1
        # This corresponds to processing bars 1 to N as specified in the problem statement.
        for i in range(N): 
            current_r = current_pos[i] # Get the current row of bar i
            
            # If bar is already at the bottom row (row H), it cannot move further down
            if current_r == H:
                continue

            # Retrieve bar i's fixed properties: column C and length L from initial data
            # bars_data is indexed 0..N-1, matching the loop variable i
            bar_info = bars_data[i] 
            C_i, L_i = bar_info['C'], bar_info['L']
            C_i_start = C_i           # Start column of bar i
            C_i_end = C_i + L_i - 1 # End column of bar i
            
            # The potential row to move to is one step below the current row
            target_r = current_r + 1
            
            # Check if the move to target_r is blocked by any bar currently present at target_r
            blocked = False
            # Check only if target_r actually has any bars
            if target_r in row_map:
                 bars_in_target_r = row_map[target_r] # Get dictionary of bars at target row
                 
                 # Iterate through all bars (bar_id_j) currently located at the target row
                 for bar_id_j, (C_j_start, C_j_end) in bars_in_target_r.items():
                     # Check for horizontal overlap between bar i's interval [C_i_start, C_i_end]
                     # and bar j's interval [C_j_start, C_j_end].
                     # Overlap occurs if the intervals intersect. A standard check for interval intersection is:
                     # max(start1, start2) <= min(end1, end2)
                     if max(C_i_start, C_j_start) <= min(C_i_end, C_j_end):
                          blocked = True # Bar i is blocked by bar j
                          break # Found an obstacle, no need to check further bars in this row
            
            # If the path down is not blocked
            if not blocked:
                 # Execute the move: bar i moves down by one row
                 
                 # Remove bar i's entry from its current row in row_map
                 # Retrieve the coordinates interval data before deleting, needed for adding to the new row dictionary entry
                 bar_coords = row_map[current_r][i] 
                 del row_map[current_r][i]
                 
                 # If the dictionary for the current row becomes empty after removal, 
                 # delete the row key itself from row_map for cleanup and potentially better performance
                 if not row_map[current_r]: 
                      del row_map[current_r]

                 # Add bar i to the target row in row_map
                 # If target row key is not yet in row_map, initialize its dictionary
                 if target_r not in row_map:
                      row_map[target_r] = {} 
                 # Add bar i with its coordinates interval (C_start, C_end) to the target row dictionary
                 row_map[target_r][i] = bar_coords 

                 # Update the current position (row) of bar i in the current_pos tracker
                 current_pos[i] = target_r
                 
                 # Since a bar moved in this pass, set the active flag to True.
                 # This signals that the simulation needs to continue for at least another pass
                 # to check if further moves are possible or if the system has stabilized.
                 active = True 

    # The simulation loop terminates when a full pass (i=0..N-1) completes with no bars moving.
    # At this point, all bars are in their final resting positions.
    
    # Prepare a list containing the final row number for each bar, ordered by bar ID 0 to N-1
    final_rows = [0] * N
    for i in range(N):
        final_rows[i] = current_pos[i]

    # Print the final row for each bar, one per line, as required by the output format
    for r in final_rows:
        print(r)

# Execute the main simulation logic
solve()