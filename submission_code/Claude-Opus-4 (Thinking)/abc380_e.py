# YOUR CODE HERE
def find_reachable(cells, x):
    """Find all cells reachable from cell x (1-indexed)"""
    n = len(cells)
    start_idx = x - 1  # Convert to 0-indexed
    color = cells[start_idx]
    
    # Find leftmost reachable cell
    left = start_idx
    while left > 0 and cells[left - 1] == color:
        left -= 1
    
    # Find rightmost reachable cell
    right = start_idx
    while right < n - 1 and cells[right + 1] == color:
        right += 1
    
    return list(range(left, right + 1))

n, q = map(int, input().split())
cells = list(range(1, n + 1))  # Initially, cell i has color i

# Initialize color count
color_count = {}
for i in range(1, n + 1):
    color_count[i] = 1

for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        x, c = query[1], query[2]
        reachable = find_reachable(cells, x)
        for pos in reachable:
            old_color = cells[pos]
            if old_color in color_count:
                color_count[old_color] -= 1
                if color_count[old_color] == 0:
                    del color_count[old_color]
            cells[pos] = c
            color_count[c] = color_count.get(c, 0) + 1
    else:  # query[0] == 2
        c = query[1]
        print(color_count.get(c, 0))