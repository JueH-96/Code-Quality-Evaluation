# YOUR CODE HERE
import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

box_colors = [set() for _ in range(N + 1)]  # Index 0 unused, boxes 1 to N

# Read initial colors for each box and add to their sets
for i in range(1, N + 1):
    color = int(data[index])
    box_colors[i].add(color)
    index += 1

# Process each query
for _ in range(Q):
    a = int(data[index])
    b = int(data[index + 1])
    index += 2
    
    # Move all colors from box a to box b
    a_set = box_colors[a]
    for color in list(a_set):  # Make a copy of the set to avoid runtime errors
        box_colors[b].add(color)
    a_set.clear()
    
    # Print the number of different colors in box b
    print(len(box_colors[b]))