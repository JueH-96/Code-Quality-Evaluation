# YOUR CODE HERE
from collections import defaultdict

# Read input
N, Q = map(int, input().split())
colors = list(map(int, input().split()))

# Initialize boxes
boxes = [set([color]) for color in colors]

for _ in range(Q):
    a, b = map(int, input().split())
    a -= 1  # Convert to 0-based indexing
    b -= 1

    # Move balls from box a to box b
    boxes[b].update(boxes[a])
    boxes[a].clear()

    # Print the number of different colors in box b
    print(len(boxes[b]))