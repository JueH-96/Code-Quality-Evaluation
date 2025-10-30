import sys

# Read all input and create an iterator
data = iter(sys.stdin.read().split())

# Read N and Q
N = int(next(data))
Q = int(next(data))

# Initialize head history with initial position after 0 moves
head_history = [(1, 0)]

# Define movement deltas
move_delta = {'R': (1, 0), 'L': (-1, 0), 'U': (0, 1), 'D': (0, -1)}

# Process each query
for _ in range(Q):
    query_type = int(next(data))
    if query_type == 1:
        # Move query
        C = next(data)  # Direction: R, L, U, D
        # Get current head position
        current_x, current_y = head_history[-1]
        # Get delta based on direction
        dx, dy = move_delta[C]
        # Calculate new position
        new_x = current_x + dx
        new_y = current_y + dy
        # Append new position to history
        head_history.append((new_x, new_y))
    elif query_type == 2:
        # Position query
        p = int(next(data))  # Part number
        # Current number of moves
        m = len(head_history) - 1
        # Calculate time t
        t = m - (p - 1)
        if t >= 0:
            # Position is in history
            x, y = head_history[t]
        else:
            # Position before first move
            x = 1 - t
            y = 0
        # Output the position
        print(f"{x} {y}")