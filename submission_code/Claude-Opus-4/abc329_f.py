# YOUR CODE HERE
N, Q = map(int, input().split())
colors = list(map(int, input().split()))

# Initialize boxes with sets to store colors
boxes = [set() for _ in range(N + 1)]  # 1-indexed, so we need N+1

# Initially, box i contains one ball of color C_i
for i in range(N):
    boxes[i + 1].add(colors[i])

# Process queries
for _ in range(Q):
    a, b = map(int, input().split())
    
    # Move all balls from box a to box b
    # This means adding all colors from box a to box b
    boxes[b].update(boxes[a])
    
    # Clear box a since all balls are moved
    boxes[a].clear()
    
    # Print the number of different colors in box b
    print(len(boxes[b]))