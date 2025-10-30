import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
Q = int(data[index + 1])
index += 2

# Initialize the boxes with their initial colors
boxes = [set() for _ in range(N + 1)]
for i in range(N):
    boxes[i + 1].add(int(data[index + i]))

index += N

# Process each query
for _ in range(Q):
    a = int(data[index])
    b = int(data[index + 1])
    index += 2

    # Move all balls from box a to box b
    boxes[b].update(boxes[a])
    boxes[a].clear()

    # Print the number of different colors in box b
    print(len(boxes[b]))