n, w = map(int, input().split())

blocks = []
for i in range(n):
    x, y = map(int, input().split())
    blocks.append((i, x-1, y-1))  # (id, col, row) 0-indexed

q = int(input())

# Current positions
pos = {i: (x, y) for i, x, y in blocks}
removed_time = {}

t = 0
# Simulate until steady state or all blocks removed
while pos and t < 10**6:  # Upper bound for safety
    t += 1
    
    # Step 1: Check if bottom row is full and remove if needed
    bottom_blocks = []
    for c in range(w):
        found = None
        for bid, (col, row) in pos.items():
            if col == c and row == 0:
                found = bid
                break
        if found is None:
            bottom_blocks = []
            break
        bottom_blocks.append(found)
    
    if len(bottom_blocks) == w:
        # Remove all blocks in bottom row
        for bid in bottom_blocks:
            removed_time[bid] = t
            del pos[bid]
    
    if not pos:
        break
    
    # Step 2: Move blocks down one cell if possible
    # Sort blocks by row (bottom to top)
    blocks_sorted = sorted(pos.items(), key=lambda x: x[1][1])
    
    new_pos = {}
    occupied = set()
    
    for bid, (col, row) in blocks_sorted:
        if row == 0 or (col, row-1) in occupied:
            # Block can't move (at bottom or blocked)
            new_pos[bid] = (col, row)
            occupied.add((col, row))
        else:
            # Block moves down one cell
            new_pos[bid] = (col, row-1)
            occupied.add((col, row-1))
    
    # Check if any change occurred
    if pos == new_pos:
        # No blocks moved - check if we can ever fill bottom row
        cols_with_blocks = set(c for _, (c, r) in pos.items())
        if len(cols_with_blocks) < w:
            # Some columns have no blocks, can't remove any more
            break
    
    pos = new_pos

# Answer queries
for _ in range(q):
    t_query, a = map(int, input().split())
    a -= 1  # Convert to 0-indexed
    
    if a in removed_time and removed_time[a] <= t_query:
        print("No")
    else:
        print("Yes")