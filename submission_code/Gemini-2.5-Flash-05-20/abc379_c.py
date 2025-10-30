import sys

def solve():
    # Read N and M from the first line
    N, M = map(int, sys.stdin.readline().split())
    
    # Read X_vals (cell numbers) and A_vals (stone counts) from the next two lines
    X_vals = list(map(int, sys.stdin.readline().split()))
    A_vals = list(map(int, sys.stdin.readline().split()))

    # Combine X_vals and A_vals into a list of (cell_number, stone_count) pairs
    # Also calculate the total number of stones initially present
    pos_stones = []
    total_initial_stones = 0
    for i in range(M):
        pos_stones.append((X_vals[i], A_vals[i]))
        total_initial_stones += A_vals[i]

    # Check if the total number of stones is equal to N (the target total)
    # If not, it's impossible to achieve the goal state.
    if total_initial_stones != N:
        print(-1)
        return

    # Sort the stone locations based on their cell number (X_i) in ascending order.
    # This is crucial for processing cells from left to right.
    pos_stones.sort()

    # `current_balance` stores the number of excess stones available
    # at the `last_processed_cell` that can be moved further to the right.
    current_balance = 0  
    # `total_operations` accumulates the sum of operations (stone movements).
    total_operations = 0
    # `last_processed_cell` tracks the last cell number (X_i) for which we have accounted for stones.
    # It starts at 0, conceptually representing a point before cell 1.
    last_processed_cell = 0 

    # Iterate through each (cell_number, stone_count) pair in sorted order
    for x, a in pos_stones:
        # Calculate the `gap_length`: the number of empty cells between `last_processed_cell` and `x`.
        # These are cells from (last_processed_cell + 1) up to (x - 1).
        gap_length = x - last_processed_cell - 1
        
        # If there is a gap (i.e., `gap_length` is positive, meaning there are cells to fill)
        if gap_length > 0:
            # Check if we have enough `current_balance` stones to fill this gap.
            # If not, it's impossible to fulfill the requirements for cells in this gap.
            if current_balance < gap_length:
                print(-1)
                return
            
            # Calculate the operations needed to move `gap_length` stones from `last_processed_cell`
            # to fill the cells in this gap.
            # The first stone moves 1 step (to last_processed_cell + 1).
            # The second stone moves 2 steps (to last_processed_cell + 2).
            # ...
            # The `gap_length`-th stone moves `gap_length` steps (to last_processed_cell + gap_length, which is x - 1).
            # The total operations for this segment is the sum of an arithmetic series: 1 + 2 + ... + gap_length.
            total_operations += gap_length * (gap_length + 1) // 2
            
            # Decrease `current_balance` as these stones have been used to fill the gap.
            current_balance -= gap_length

        # At the current cell `x`:
        # Add the `a` stones initially present at `x` to `current_balance`.
        current_balance += a
        
        # One stone is needed for cell `x` itself. This stone is "consumed" from the `current_balance`.
        current_balance -= 1 
        
        # After processing cell `x` (adding its stones and consuming one),
        # if `current_balance` becomes negative, it means we didn't have enough stones
        # to fill all cells from 1 up to `x` (even with stones at `x`). This is an impossible state.
        if current_balance < 0:
            print(-1)
            return

        # Update `last_processed_cell` to the current cell `x`.
        last_processed_cell = x

    # After iterating through all initial stone locations, there might be a remaining gap
    # from the `last_processed_cell` to the final cell `N`.
    remaining_gap_length = N - last_processed_cell
    
    # If such a remaining gap exists
    if remaining_gap_length > 0:
        # Check if there are enough excess stones (current_balance) to fill this final gap.
        if current_balance < remaining_gap_length:
            print(-1)
            return
        
        # Calculate operations for this final gap using the same arithmetic series logic.
        total_operations += remaining_gap_length * (remaining_gap_length + 1) // 2
        # `current_balance` would become 0 after this, but we don't need to update it further.

    # Print the total minimum operations required.
    print(total_operations)

# Call the solve function to execute the program
solve()