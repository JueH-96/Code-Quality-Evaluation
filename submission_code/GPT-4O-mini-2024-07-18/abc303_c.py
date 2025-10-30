def can_complete_moves(N, M, H, K, S, items):
    # Set to store item locations for quick access
    item_set = set(items)
    
    # Starting position and health
    x, y = 0, 0
    health = H
    
    # Process each move
    for move in S:
        # Deduct health for the move
        health -= 1
        
        # Check if health is negative
        if health < 0:
            return "No"
        
        # Update position based on the move
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        
        # Check if there's an item at the new position
        if (x, y) in item_set and health < K:
            health = K  # Consume the item and restore health to K
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

# First line contains N, M, H, K
N, M, H, K = map(int, data[0].split())
# Second line contains the string S
S = data[1]
# Next M lines contain the item positions
items = [tuple(map(int, line.split())) for line in data[2:2 + M]]

# Get the result
result = can_complete_moves(N, M, H, K, S, items)

# Print the result
print(result)