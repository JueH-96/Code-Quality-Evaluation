def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO approach.
    # Parse inputs
    H, W, Q = map(int, input_data[:3])
    queries = input_data[3:]
    
    # Number of cells
    n = H * W
    
    # We will maintain for each cell (in 0-based index):
    #   rowPrev[x], rowNext[x]: the neighboring cell in the same row (left/right)
    #   colPrev[x], colNext[x]: the neighboring cell in the same column (up/down)
    # A value of -1 indicates "none" (edge of row/column or removed).
    #
    # "destroyed[x]" indicates whether cell x is already removed (its wall destroyed).
    # We also maintain a counter for how many walls remain.
    
    # If idx = (r-1)*W + (c-1), r in [1..H], c in [1..W],
    # then the cell is at row r, column c in 1-based coordinates.
    # rowPrev[x], rowNext[x] link horizontally within row r
    # colPrev[x], colNext[x] link vertically within column c
    
    # To remove a cell x:
    #   1) If rowPrev[x] != -1, link its row-prev to its row-next
    #   2) If rowNext[x] != -1, link its row-next to its row-prev
    #   3) If colPrev[x] != -1, link its col-prev to its col-next
    #   4) If colNext[x] != -1, link its col-next to its col-prev
    #   5) destroyed[x] = True
    #   6) decrement the total count of walls
    
    sys.setrecursionlimit(10**7)
    
    # Helper to convert (r, c) -> index
    def idx(r, c):
        return (r - 1) * W + (c - 1)
    
    # Build the linked-lists for rows and columns.
    rowPrev = [-1] * n
    rowNext = [-1] * n
    colPrev = [-1] * n
    colNext = [-1] * n
    destroyed = [False] * n
    
    # Initialize row links
    # For row i in [1..H], the cells in that row are from i*W + 0.. i*W+ (W-1) in 0-based,
    # but let's do it by (i, j) => idx = (i-1)*W + (j-1).
    # rowNext of cell => next column, rowPrev => previous column
    # colNext of cell => next row, colPrev => previous row
    # We'll do a double pass: one for row links, one for col links.
    
    # Row links
    for r in range(1, H + 1):
        start = (r - 1) * W
        # The cells in row r in 0-based are [start, start+1, ..., start+(W-1)]
        for j in range(W):
            x = start + j
            # rowPrev
            if j == 0:
                rowPrev[x] = -1
            else:
                rowPrev[x] = x - 1
            # rowNext
            if j == W - 1:
                rowNext[x] = -1
            else:
                rowNext[x] = x + 1
    
    # Column links
    for c in range(1, W + 1):
        # The cells in column c are (1,c), (2,c), ..., (H,c).
        # in 0-based index => idx(1,c), idx(2,c), ...
        # that is [ (0)*W + (c-1), (1)*W + (c-1), ..., (H-1)*W + (c-1) ]
        start = c - 1
        for r in range(H):
            x = r * W + start
            # colPrev
            if r == 0:
                colPrev[x] = -1
            else:
                colPrev[x] = x - W
            # colNext
            if r == H - 1:
                colNext[x] = -1
            else:
                colNext[x] = x + W
    
    # Count of remaining walls
    remain = n
    
    # Helper to remove a cell (destroy its wall) if not already destroyed
    def remove_cell(x):
        nonlocal remain
        if destroyed[x]:
            return
        destroyed[x] = True
        remain -= 1
        
        # Unlink from row
        lp = rowPrev[x]
        ln = rowNext[x]
        if lp != -1:
            rowNext[lp] = ln
        if ln != -1:
            rowPrev[ln] = lp
        
        # Unlink from column
        up = colPrev[x]
        dn = colNext[x]
        if up != -1:
            colNext[up] = dn
        if dn != -1:
            colPrev[dn] = up
    
    # Process the queries
    pos = 0
    output = []
    for _ in range(Q):
        Rq = int(queries[pos]); pos += 1
        Cq = int(queries[pos]); pos += 1
        qidx = idx(Rq, Cq)
        
        # If there's a wall at (Rq, Cq), destroy it and end process.
        if not destroyed[qidx]:
            remove_cell(qidx)
        else:
            # If no wall at (Rq, Cq), destroy the first walls up, down, left, right
            up = colPrev[qidx]
            down = colNext[qidx]
            left = rowPrev[qidx]
            right = rowNext[qidx]
            
            if up != -1:
                remove_cell(up)
            if down != -1:
                remove_cell(down)
            if left != -1:
                remove_cell(left)
            if right != -1:
                remove_cell(right)
    
    # Print the number of remaining walls
    print(remain)

# Call solve()
solve()