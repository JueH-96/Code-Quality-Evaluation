from collections import defaultdict

# Read the number of boxes and queries
N, Q = map(int, input().split())

# Read the initial colors of the balls in the boxes
colors = list(map(int, input().split()))

# Initialize the boxes with their respective colors
boxes = defaultdict(set)
for i, color in enumerate(colors, start=1):
    boxes[i].add(color)

# Process each query
for _ in range(Q):
    a, b = map(int, input().split())
    
    # Move all balls from box a to box b
    boxes[b].update(boxes[a])
    boxes[a].clear()
    
    # Print the number of different colors in box b
    print(len(boxes[b]))