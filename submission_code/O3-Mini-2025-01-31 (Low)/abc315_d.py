def main():
    import sys
    from collections import deque
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]
    
    # For each row i, rowCells[i] is a set of column indices (j) where the cookie is still present.
    rowCells = [set(range(W)) for _ in range(H)]
    # Similarly, for each column j, colCells[j] is a set of row indices (i) where the cookie is still present.
    colCells = [set(range(H)) for _ in range(W)]
    
    # Functions to check if a row (or column) is homogeneous:
    def check_row(i):
        cells = rowCells[i]
        if len(cells) < 2:
            return False
        # Get an arbitrary column in the row.
        it = iter(cells)
        first = next(it)
        base = grid[i][first]
        # If every cookie in the row (that still exists) is base, the row is homogeneous.
        for j in cells:
            if grid[i][j] != base:
                return False
        return True

    def check_col(j):
        cells = colCells[j]
        if len(cells) < 2:
            return False
        it = iter(cells)
        first = next(it)
        base = grid[first][j]
        for i in cells:
            if grid[i][j] != base:
                return False
        return True

    # Use deques to hold candidates rows and columns that might be removed this round.
    rowQueue = deque()
    colQueue = deque()
    
    # Initialize the queues by checking every row and column.
    for i in range(H):
        if check_row(i):
            rowQueue.append(i)
    for j in range(W):
        if check_col(j):
            colQueue.append(j)
    
    # Repeat rounds until no row or column qualifies.
    while rowQueue or colQueue:
        # To simulate the procedure's round, we first gather all cookies (cells) that are marked.
        toRemove = set()
        # Process rows: if a row is homogeneous, mark all its cookies.
        while rowQueue:
            i = rowQueue.popleft()
            if check_row(i):  # Re-check since states might have changed.
                for j in rowCells[i]:
                    toRemove.add((i, j))
        # Process columns similarly.
        while colQueue:
            j = colQueue.popleft()
            if check_col(j):
                for i in colCells[j]:
                    toRemove.add((i, j))
                    
        # If no cookies were marked in this round, the procedure terminates.
        if not toRemove:
            break
        
        # When cookies are removed, update the rowCells and colCells sets.
        affectedRows = set()
        affectedCols = set()
        for (i, j) in toRemove:
            # Remove cookie (i,j) if it has not been removed already.
            if j in rowCells[i]:
                rowCells[i].remove(j)
                affectedRows.add(i)
            if i in colCells[j]:
                colCells[j].remove(i)
                affectedCols.add(j)
        
        # Any row or column affected by removals may now become homogeneous.
        for i in affectedRows:
            if check_row(i):
                rowQueue.append(i)
        for j in affectedCols:
            if check_col(j):
                colQueue.append(j)
                
    # Finally, count the remaining cookies.
    remaining = 0
    for cells in rowCells:
        remaining += len(cells)
    print(remaining)
    
if __name__ == "__main__":
    main()