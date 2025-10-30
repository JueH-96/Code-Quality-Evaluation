import sys

def solve():
    # Read N, R, C from the first line
    N, R, C = map(int, sys.stdin.readline().split())
    # Read the wind string S from the second line
    S = sys.stdin.readline().strip()

    # Define displacement vectors for each direction
    # Based on the problem description:
    # N: (r-1,c) -> (dr, dc) = (-1, 0)
    # W: (r,c-1) -> (dr, dc) = (0, -1)
    # S: (r+1,c) -> (dr, dc) = (1, 0)
    # E: (r,c+1) -> (dr, dc) = (0, 1)
    move_map = {
        'N': (-1, 0),
        'W': (0, -1),
        'S': (1, 0),
        'E': (0, 1)
    }

    # Use a set to store the coordinates (r, c) where smoke exists.
    # A set provides O(1) average time complexity for add, remove, and check operations,
    # which is crucial for performance given N can be up to 200,000.
    # Initially, smoke only exists at (0,0).
    current_smoke_positions = set()
    current_smoke_positions.add((0, 0))

    # List to store the '0' or '1' for each time step's result
    result_output = []

    # Simulate the process for each time step from t=1 to N
    for i in range(N):
        # Get the displacement (dr, dc) for the current wind character S[i]
        dr, dc = move_map[S[i]]

        # Step 1: Wind blows - all existing smoke moves
        next_smoke_positions = set()
        # Flag to check if the (0,0) cell is covered after smoke moves due to wind.
        # This is important for the next step (smoke generation).
        campfire_covered_by_wind = False 
        
        for r, c in current_smoke_positions:
            new_r, new_c = r + dr, c + dc
            next_smoke_positions.add((new_r, new_c))
            
            # Check if the moved smoke particle landed on (0,0)
            if new_r == 0 and new_c == 0:
                campfire_covered_by_wind = True

        # Step 2: If there is no smoke in cell (0,0) AFTER wind blows, 
        # new smoke is generated at (0,0).
        if not campfire_covered_by_wind:
            next_smoke_positions.add((0, 0))
        
        # Update the set of smoke positions for the next time step
        current_smoke_positions = next_smoke_positions

        # Determine if smoke exists at cell (R,C) at time (i+1)+0.5
        # This is after all operations for time step (i+1) are complete.
        if (R, C) in current_smoke_positions:
            result_output.append('1')
        else:
            result_output.append('0')
    
    # Print the N-character string joined from the result list
    sys.stdout.write("".join(result_output) + "
")

# Call the solve function to run the program
solve()