N, Q = map(int, input().split())
colors = list(map(int, input().split()))

# Initialize boxes as list of sets to store colors in each box
boxes = [set([c]) for c in colors]

for _ in range(Q):
    a, b = map(int, input().split())
    a -= 1  # Convert to 0-based indexing
    b -= 1
    
    # Move all balls from box a to box b
    boxes[b].update(boxes[a])
    boxes[a].clear()
    
    # Print number of different colors in box b
    print(len(boxes[b]))