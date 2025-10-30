# Read input
N, Q = map(int, input().split())
colors = list(map(int, input().split()))

# Initialize boxes with given colors
boxes = [set() for _ in range(N + 1)]
for i in range(1, N + 1):
    boxes[i].add(colors[i - 1])

# Process queries
for _ in range(Q):
    a, b = map(int, input().split())
    boxes[b].update(boxes[a])  # Move all colors from box a to box b
    boxes[a].clear()  # Empty box a
    print(len(boxes[b]))  # Print number of distinct colors in box b