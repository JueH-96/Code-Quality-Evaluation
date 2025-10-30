import sys

def solve():
    # Read N, M, H, K from the first line
    N, M, H, K = map(int, sys.stdin.readline().split())

    # Read the move sequence string S
    S = sys.stdin.readline().strip()

    # Read M item locations and store them in a set for efficient lookup
    # A set allows O(1) average-case time complexity for checking if a point is an item location.
    item_locations = set()
    for _ in range(M):
        x_item, y_item = map(int, sys.stdin.readline().split())
        item_locations.add((x_item, y_item))

    # Initialize Takahashi's current position and health
    current_x, current_y = 0, 0
    current_health = H

    # Simulate each of the N moves
    for move_char in S:
        # 1. Consume health for movement
        current_health -= 1

        # 2. Check for collapse: If health drops below 0, Takahashi collapses.
        if current_health < 0:
            print("No")
            return # Exit immediately as he cannot complete the moves

        # 3. Update coordinates based on the current move character
        if move_char == 'R':
            current_x += 1
        elif move_char == 'L':
            current_x -= 1
        elif move_char == 'U':
            current_y += 1
        elif move_char == 'D':
            current_y -= 1
        
        # 4. Check for item consumption at the new location
        # An item is consumed if:
        #   a) An item is placed at the current coordinates.
        #   b) Takahashi's current health is strictly less than K.
        # If both conditions are met, his health becomes K.
        # Note: Items are reusable; they are not removed from `item_locations` after use.
        if (current_x, current_y) in item_locations:
            if current_health < K:
                current_health = K

    # If the loop completes, it means Takahashi successfully made all N moves
    # without his health dropping below zero.
    print("Yes")

# Call the solve function to execute the program
solve()