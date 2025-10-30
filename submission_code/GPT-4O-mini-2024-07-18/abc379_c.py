def min_operations_to_distribute_stones(N, M, positions, amounts):
    # Total stones available
    total_stones = sum(amounts)
    
    # Check if we can even reach the target of N stones (1 in each cell)
    if total_stones < N:
        return -1
    
    # Create a list to hold the number of stones in each position
    stones = [0] * (N + 1)
    
    # Fill the stones in the respective positions
    for pos, amt in zip(positions, amounts):
        stones[pos] = amt
    
    # We need to keep track of the number of moves
    moves = 0
    # We will use a variable to carry over excess stones
    carry_over = 0
    
    # Iterate through each cell from 1 to N
    for i in range(1, N + 1):
        # Add carry over from previous cells
        carry_over += stones[i]
        
        # We need exactly 1 stone in each cell
        if carry_over < 1:
            return -1  # Not enough stones to satisfy this cell
        
        # Calculate how many stones we need to move out of this cell
        # We can only keep 1 stone in this cell
        moves += carry_over - 1
        
        # After satisfying this cell, we have carry_over - 1 stones left
        carry_over = carry_over - 1
    
    return moves

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
positions = list(map(int, data[2:2 + M]))
amounts = list(map(int, data[2 + M:2 + 2 * M]))

# Get the result
result = min_operations_to_distribute_stones(N, M, positions, amounts)

# Print the result
print(result)