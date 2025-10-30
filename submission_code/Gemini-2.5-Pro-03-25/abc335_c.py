# YOUR CODE HERE
import sys

# Function to read input line faster
# Using sys.stdin.readline is generally faster than input() for competitive programming
input = sys.stdin.readline

def solve():
    # Read the number of parts N and the number of queries Q.
    # N is part of the problem context (defines the range for p) but is not directly
    # used in the calculation logic of the optimized approach.
    N, Q = map(int, input().split())

    # head_history stores the sequence of positions of the head (part 1).
    # head_history[t] stores a tuple (x, y) representing the head's position 
    # after t moves (type 1 queries have been processed).
    # We initialize it with the position after 0 moves, P_0 = (1, 0).
    # A Python list provides efficient O(1) indexing and amortized O(1) append operations.
    head_history = [(1, 0)] 
    
    # k tracks the total number of type 1 queries (moves) processed so far.
    # It starts at 0.
    k = 0 

    # Process all Q queries one by one
    for _ in range(Q):
        # Read the current query line and split it into parts
        query_parts = input().split()
        # The first part indicates the query type (1 or 2)
        query_type = int(query_parts[0])

        if query_type == 1:
            # Type 1 query: Move the head
            
            # Increment the move counter because a type 1 query represents one step/move.
            k += 1 
            
            # The second part of the query specifies the direction of movement.
            direction = query_parts[1]
            
            # Get the most recent head position from the history list.
            # This is the position after k-1 moves. head_history[-1] gives the last element.
            cx, cy = head_history[-1]
            
            # Calculate the new head position based on the direction.
            # Start with the current position and modify coordinates based on 'R', 'L', 'U', 'D'.
            nx, ny = cx, cy
            if direction == 'R':
                nx += 1 # Move right: increase x by 1
            elif direction == 'L':
                nx -= 1 # Move left: decrease x by 1
            elif direction == 'U':
                ny += 1 # Move up: increase y by 1
            elif direction == 'D':
                ny -= 1 # Move down: decrease y by 1
            
            # Append the newly calculated head position (position after k moves) to the history list.
            head_history.append((nx, ny))

        else: # query_type == 2
            # Type 2 query: Report the position of part p
            
            # The second part of the query gives the part number p.
            p = int(query_parts[1])
            
            # Determine the position of part p after k total moves.
            # The core logic derived is that the position of part p after k moves is:
            # - The position of the head after k - (p - 1) moves, if k >= p - 1.
            # - The position (p - k, 0), if k < p - 1.
            
            if k >= p - 1:
                # If k >= p - 1, part p has moved enough steps to be located at a position 
                # previously occupied by the head. Specifically, it's at the position the head
                # occupied after k - p + 1 moves.
                # The index in the 0-based list head_history is k - p + 1.
                idx = k - p + 1
                pos = head_history[idx]
                # Print the coordinates x and y separated by a space.
                print(f"{pos[0]} {pos[1]}")
            else:
                # If k < p - 1, part p has not yet completed p-1 moves. It is still effectively 
                # moving along the initial configuration line on the x-axis.
                # It started at (p, 0). After k steps following the part ahead, 
                # it reaches the position that part p-k occupied initially.
                # This initial position was (p - k, 0).
                # Print the coordinates x and y separated by a space.
                print(f"{p - k} 0")

# Execute the main solution function when the script is run.
solve()

# END OF YOUR CODE HERE