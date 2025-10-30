# YOUR CODE HERE
import sys

def solve():
    """
    Solves the Takahashi's Moves problem by simulating his journey.
    """
    # Read problem parameters from standard input using fast I/O.
    try:
        line1 = sys.stdin.readline()
        # Handle empty input at the end of file
        if not line1:
            return
        N, M, H, K = map(int, line1.split())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        return

    # Use a set comprehension for a concise and efficient way to read
    # and store the item locations. A set provides O(1) average time
    # for lookups and removals.
    items = {tuple(map(int, sys.stdin.readline().split())) for _ in range(M)}

    # Initialize Takahashi's state: position (x, y) and health.
    x, y = 0, 0
    health = H

    # Simulate the N moves, one for each character in the move string S.
    for move in S:
        # Step 1: Each move costs 1 health point.
        health -= 1

        # Step 2: Update position based on the move character.
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        else:  # move == 'D'
            y -= 1
    
        # Step 3: Check if health has dropped below zero. If so, he collapses.
        if health < 0:
            print("No")
            # Terminate the program as the outcome is determined.
            return

        # Step 4: Check if an item can be consumed.
        # This check is performed only if health is below the threshold K.
        # Short-circuiting `and` makes this efficient.
        if health < K and (x, y) in items:
            # Restore health to K.
            health = K
            # Remove the item from the set as it is now consumed.
            items.remove((x, y))

    # If the loop completes, Takahashi has survived all N moves.
    print("Yes")

# Run the solution
solve()