# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    # Initialize presence matrix
    present = [[True] * W for _ in range(H)]
    
    # Initialize row and column counters
    row_counters = [defaultdict(int) for _ in range(H)]
    col_counters = [defaultdict(int) for _ in range(W)]
    row_counts = [0] * H
    col_counts = [0] * W
    
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            row_counters[r][color] += 1
            col_counters[c][color] += 1
            row_counts[r] +=1
            col_counts[c] +=1
            
    while True:
        marked_rows = set()
        marked_cols = set()
        
        # Find rows to mark
        for r in range(H):
            if row_counts[r] >=2 and len(row_counters[r]) ==1:
                marked_rows.add(r)
        
        # Find columns to mark
        for c in range(W):
            if col_counts[c] >=2 and len(col_counters[c]) ==1:
                marked_cols.add(c)
        
        if not marked_rows and not marked_cols:
            break
        
        # To avoid double removal, mark cells from rows and columns
        to_remove = set()
        for r in marked_rows:
            for c in range(W):
                if present[r][c]:
                    to_remove.add((r, c))
        for c in marked_cols:
            for r in range(H):
                if present[r][c]:
                    to_remove.add((r, c))
        
        if not to_remove:
            break
        
        # Remove marked cells and update counters
        for r, c in to_remove:
            if present[r][c]:
                present[r][c] = False
                color = grid[r][c]
                row_counters[r][color] -=1
                if row_counters[r][color] ==0:
                    del row_counters[r][color]
                row_counts[r] -=1
                
                col_counters[c][color] -=1
                if col_counters[c][color] ==0:
                    del col_counters[c][color]
                col_counts[c] -=1
    
    # Count remaining cookies
    remaining = 0
    for r in range(H):
        remaining += row_counts[r]
    print(remaining)

if __name__ == "__main__":
    main()