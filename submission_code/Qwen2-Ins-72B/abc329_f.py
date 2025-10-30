from collections import defaultdict

N, Q = map(int, input().split())
C = list(map(int, input().split()))

# Initialize data structures
box_contents = defaultdict(set)
box_colors = [set([C[i]]) for i in range(N)]
color_boxes = defaultdict(set)
for i in range(N):
    color_boxes[C[i]].add(i)

for _ in range(Q):
    a, b = map(int, input().split())
    a -= 1
    b -= 1

    # Move balls from box a to box b
    for color in box_colors[a]:
        box_contents[b].add(color)
        color_boxes[color].remove(a)
        if a in color_boxes[color]:
            color_boxes[color].remove(a)
        color_boxes[color].add(b)

    # Update box_colors
    box_colors[b] = box_contents[b].copy()
    box_contents[a].clear()
    box_colors[a].clear()

    # Print the number of different colors in box b
    print(len(box_colors[b]))