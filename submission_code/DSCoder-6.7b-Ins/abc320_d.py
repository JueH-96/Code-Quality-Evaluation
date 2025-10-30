import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    info = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]

    # Create a dictionary to store the relative positions
    relative_positions = defaultdict(list)
    for a, b, x, y in info:
        relative_positions[(a, b)].append((x, y))

    # Create a dictionary to store the absolute positions
    absolute_positions = defaultdict(list)
    for (a, b), positions in relative_positions.items():
        # If there are multiple positions, they must be the same
        if len(set(positions)) > 1:
            print("undecidable")
            return
        # The absolute position is the sum of the relative positions
        x, y = positions[0]
        absolute_positions[a].append((x, y))

    # If there are multiple absolute positions, they must be the same
    for person, positions in absolute_positions.items():
        if len(set(positions)) > 1:
            print("undecidable")
            return

    # Print the absolute positions
    for person in range(1, N+1):
        if person in absolute_positions:
            x, y = absolute_positions[person][0]
            print(x, y)
        else:
            print("undecidable")

solve()