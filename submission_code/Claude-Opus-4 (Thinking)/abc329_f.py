N, Q = map(int, input().split())
C = list(map(int, input().split()))

# Initialize boxes with sets to track distinct colors
boxes = [set() for _ in range(N + 1)]  # 0 index is unused
for i in range(N):
    boxes[i + 1].add(C[i])

for _ in range(Q):
    a, b = map(int, input().split())
    
    # Move all balls from box a to box b
    boxes[b] |= boxes[a]  # Add all colors from box a to box b
    boxes[a].clear()      # Empty box a
    
    # Print the number of different colors in box b
    print(len(boxes[b]))