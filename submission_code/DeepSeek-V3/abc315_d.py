# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    # Initialize row and column counts
    row_counts = [defaultdict(int) for _ in range(H)]
    col_counts = [defaultdict(int) for _ in range(W)]
    
    # Initialize remaining rows and columns
    remaining_rows = set(range(H))
    remaining_cols = set(range(W))
    
    # Initialize the counts
    for i in range(H):
        for j in range(W):
            row_counts[i][grid[i][j]] += 1
            col_counts[j][grid[i][j]] += 1
    
    while True:
        # Determine rows to remove
        rows_to_remove = set()
        for i in remaining_rows:
            if len(row_counts[i]) == 1:
                color = next(iter(row_counts[i]))
                if row_counts[i][color] >= 2:
                    rows_to_remove.add(i)
        
        # Determine columns to remove
        cols_to_remove = set()
        for j in remaining_cols:
            if len(col_counts[j]) == 1:
                color = next(iter(col_counts[j]))
                if col_counts[j][color] >= 2:
                    cols_to_remove.add(j)
        
        if not rows_to_remove and not cols_to_remove:
            break
        
        # Remove the rows
        for i in rows_to_remove:
            remaining_rows.discard(i)
            for j in remaining_cols:
                col_counts[j][grid[i][j]] -= 1
                if col_counts[j][grid[i][j]] == 0:
                    del col_counts[j][grid[i][j]]
        
        # Remove the columns
        for j in cols_to_remove:
            remaining_cols.discard(j)
            for i in remaining_rows:
                row_counts[i][grid[i][j]] -= 1
                if row_counts[i][grid[i][j]] == 0:
                    del row_counts[i][grid[i][j]]
    
    # Count the remaining cookies
    count = 0
    for i in remaining_rows:
        for j in remaining_cols:
            count += 1
    print(count)

if __name__ == "__main__":
    main()