# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read input values N, M, H, K from the first line
    N, M, H, K = map(int, sys.stdin.readline().split())
    
    # Read the movement string S from the second line
    S = sys.stdin.readline().strip()
    
    # Read M item locations and store them in a set for efficient lookup
    # Using a set allows O(1) average time complexity for checking membership and removal.
    item_locations = set()
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        item_locations.add((x, y))
        
    # Initialize Takahashi's position and health
    current_x = 0
    current_y = 0
    current_H = H
    
    # Dictionary to map movement characters to coordinate changes (dx, dy)
    move_map = {
        'R': (1, 0),
        'L': (-1, 0),
        'U': (0, 1),
        'D': (0, -1),
    }
    
    # Simulate each move as described in the problem
    for move_char in S:
        # 1. Update position based on the move character
        dx, dy = move_map[move_char]
        current_x += dx
        current_y += dy
        
        # 2. Decrease health for the move
        current_H -= 1
        
        # 3. Check if health has become negative. If so, Takahashi collapses.
        # This check happens immediately after health decreases for the move cost.
        if current_H < 0:
            print("No")
            return # Stop simulation and exit the function
        
        # 4. If Takahashi has not collapsed (current_H >= 0), check for item consumption.
        # An item is consumed *only if*:
        # a) There is an item at the current location AND
        # b) His current health is strictly less than K.
        # The check `current_pos in item_locations` is efficient with a set.
        current_pos = (current_x, current_y)
        
        if current_pos in item_locations and current_H < K:
            # Consume the item: health becomes K
            current_H = K
            # Remove the consumed item from the set so it's not used again
            item_locations.remove(current_pos)
            
    # If the loop completes all N moves without health ever dropping below zero,
    # Takahashi can complete the moves without being stunned.
    print("Yes")
    
# Call the solve function to run the simulation
solve()