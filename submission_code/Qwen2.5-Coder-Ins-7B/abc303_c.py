# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
H = int(data[2])
K = int(data[3])
S = data[4]
items = [(int(data[5 + 2 * i]), int(data[6 + 2 * i])) for i in range(M)]

# Initialize Takahashi's position and health
x, y = 0, 0
health = H

# Dictionary to store items by their positions
item_dict = {item: True for item in items}

# Process each move
for move in S:
    if move == 'R':
        x += 1
    elif move == 'L':
        x -= 1
    elif move == 'U':
        y += 1
    elif move == 'D':
        y -= 1
    
    # Reduce health by 1 for each move
    health -= 1
    
    # Check if Takahashi collapses
    if health < 0:
        print("No")
        sys.exit(0)
    
    # Check if there is an item at the current position and Takahashi's health is less than K
    if (x, y) in item_dict and health < K:
        health = K
        # Remove the item from the dictionary
        del item_dict[(x, y)]

# If all moves are completed without collapsing
print("Yes")