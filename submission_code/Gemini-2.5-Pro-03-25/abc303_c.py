# YOUR CODE HERE
import sys

# Function implementing the core logic
def solve():
    # Read first line: N, M, H, K
    # N: number of moves
    # M: number of items
    # H: initial health
    # K: health threshold for item consumption / health value after consumption
    N, M, H, K = map(int, sys.stdin.readline().split())
    
    # Read second line: move sequence string S
    S = sys.stdin.readline().strip()
    
    # Read M lines for item locations and store them in a set
    # Using a set allows for efficient checking of item existence (average O(1))
    # and removal of consumed items (average O(1)).
    # Coordinates are stored as tuples (x, y).
    items = set()
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        items.add((x, y))

    # Initialize Takahashi's current coordinates (cx, cy) starting at (0,0)
    # Initialize Takahashi's current health (ch) to initial health H
    cx, cy = 0, 0
    ch = H

    # Simulate Takahashi's N moves, indexed from 0 to N-1
    for i in range(N):
        # Get the direction for the current (i-th) move from the string S
        move = S[i]
        
        # Update coordinates based on the move direction
        if move == 'R':
            cx += 1 # Move Right: increment x coordinate
        elif move == 'L':
            cx -= 1 # Move Left: decrement x coordinate
        elif move == 'U':
            cy += 1 # Move Up: increment y coordinate
        elif move == 'D':
            cy -= 1 # Move Down: decrement y coordinate
        
        # Each move consumes 1 health point
        ch -= 1
        
        # Check if Takahashi's health drops below zero (becomes negative)
        if ch < 0:
            # If health is negative, Takahashi collapses. Print "No" and terminate the program.
            print("No")
            return # Exit the function immediately

        # Check if an item can be consumed at the new location (cx, cy)
        # Conditions for consumption:
        # 1. An item exists at the current coordinates (cx, cy). Checked using `(cx, cy) in items`.
        # 2. Takahashi's current health 'ch' is strictly less than K. Checked using `ch < K`.
        # This check happens *after* the health reduction from the move and *after* the check for collapse.
        if (cx, cy) in items and ch < K:
             # If both conditions are met, Takahashi consumes the item:
             # Restore health to the value K
             ch = K
             # Remove the consumed item from the set `items` so it cannot be used again
             # The `remove` method is safe here because we already confirmed `(cx, cy)` is in the set via the `in` check.
             items.remove((cx, cy))

    # If the loop completes all N moves without Takahashi collapsing
    # Print "Yes" indicating he successfully completed all moves.
    print("Yes")

# Call the main function `solve()` to execute the program logic.
# This reads input from stdin, runs the simulation, and prints output to stdout.
solve()