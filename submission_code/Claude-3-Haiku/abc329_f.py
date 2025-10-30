from collections import defaultdict

# Read input
N, Q = map(int, input().split())
colors = list(map(int, input().split()))

# Initialize a dictionary to store the balls in each box
boxes = defaultdict(list)
for i in range(N):
    boxes[i+1].append(colors[i])

# Process queries
for _ in range(Q):
    a, b = map(int, input().split())
    
    # Move balls from box a to box b
    boxes[b].extend(boxes[a])
    boxes[a] = []
    
    # Count the number of unique colors in box b
    print(len(set(boxes[b])))