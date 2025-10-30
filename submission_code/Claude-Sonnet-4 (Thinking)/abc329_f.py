n, q = map(int, input().split())
colors = list(map(int, input().split()))

# Initialize boxes with their initial balls (using sets for distinct colors)
boxes = [set() for _ in range(n + 1)]  # 1-indexed
for i in range(n):
    boxes[i + 1].add(colors[i])

for _ in range(q):
    a, b = map(int, input().split())
    # Move all balls from box a to box b
    boxes[b].update(boxes[a])
    boxes[a].clear()
    # Print the number of different colors in box b
    print(len(boxes[b]))