def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    # The cookie grid (list of strings)
    lines = input_data[2:]
    
    # Grid of characters
    grid = lines  # grid[i] is the string for row i
    
    # Prepare data structures
    # rowColors[i][c] = how many cookies of color c remain in row i
    # colColors[j][c] = how many cookies of color c remain in column j
    # We map 'a'..'z' to 0..25
    rowColors = [[0]*26 for _ in range(H)]
    colColors = [[0]*26 for _ in range(W)]
    
    # Keep track of how many cookies a row / column currently has
    rowCount = [W]*H
    colCount = [H]*W
    
    # Keep track of how many distinct colors are in each row/column
    rowDistinctColors = [0]*H
    colDistinctColors = [0]*W
    
    # Boolean 2D array to mark removed cookies
    removed = [[False]*W for _ in range(H)]
    
    # Initialize rowColors, colColors
    for i in range(H):
        for j in range(W):
            c_idx = ord(grid[i][j]) - ord('a')
            rowColors[i][c_idx] += 1
            colColors[j][c_idx] += 1
    
    # Compute distinct colors for rows
    for i in range(H):
        distinct = 0
        for c in range(26):
            if rowColors[i][c] > 0:
                distinct += 1
        rowDistinctColors[i] = distinct
    
    # Compute distinct colors for columns
    for j in range(W):
        distinct = 0
        for c in range(26):
            if colColors[j][c] > 0:
                distinct += 1
        colDistinctColors[j] = distinct
    
    # Total number of cookies initially
    total = H * W
    
    # We only need to re-check rows/columns that have changed
    changedRows = set(range(H))
    changedCols = set(range(W))
    
    while True:
        # Mark which rows/columns are homogeneous
        rowMarked = []
        for i in changedRows:
            # Homogeneous row => rowCount[i] >= 2 and exactly 1 distinct color
            if rowCount[i] >= 2 and rowDistinctColors[i] == 1:
                rowMarked.append(i)
        
        colMarked = []
        for j in changedCols:
            # Homogeneous column => colCount[j] >= 2 and exactly 1 distinct color
            if colCount[j] >= 2 and colDistinctColors[j] == 1:
                colMarked.append(j)
        
        # If none marked, we are done
        if not rowMarked and not colMarked:
            break
        
        # Gather cookies to remove
        # We remove all cookies in these homogeneous rows/columns in one batch
        removedCookies = []
        
        # Rows to remove
        for i in rowMarked:
            # Double-check still homogeneous (can change if rowCount < 2 after prior removals)
            if rowCount[i] >= 2 and rowDistinctColors[i] == 1:
                for j in range(W):
                    if not removed[i][j]:
                        removedCookies.append((i, j))
        
        # Columns to remove
        for j in colMarked:
            # Double-check still homogeneous
            if colCount[j] >= 2 and colDistinctColors[j] == 1:
                for i in range(H):
                    if not removed[i][j]:
                        removedCookies.append((i, j))
        
        # Remove duplicates
        to_remove_set = set(removedCookies)
        
        # Next iteration we only need to re-check rows/columns that lost cookies
        changedRowsNext = set()
        changedColsNext = set()
        
        # Perform the batch removal
        for (i, j) in to_remove_set:
            if not removed[i][j]:
                removed[i][j] = True
                c_idx = ord(grid[i][j]) - ord('a')
                
                # Update row data
                rowColors[i][c_idx] -= 1
                if rowColors[i][c_idx] == 0:
                    rowDistinctColors[i] -= 1
                rowCount[i] -= 1
                
                # Update column data
                colColors[j][c_idx] -= 1
                if colColors[j][c_idx] == 0:
                    colDistinctColors[j] -= 1
                colCount[j] -= 1
                
                total -= 1
                
                changedRowsNext.add(i)
                changedColsNext.add(j)
        
        # Update changed sets
        changedRows = changedRowsNext
        changedCols = changedColsNext
    
    print(total)

# Do not forget to call main()
if __name__ == "__main__":
    main()