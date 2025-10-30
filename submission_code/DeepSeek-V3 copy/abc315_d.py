# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    # Initialize row and column counts
    row_counts = [defaultdict(int) for _ in range(H)]
    col_counts = [defaultdict(int) for _ in range(W)]
    
    # Initialize remaining cookies
    remaining = [[True for _ in range(W)] for _ in range(H)]
    
    # Initialize row and column sizes
    row_size = [W for _ in range(H)]
    col_size = [H for _ in range(W)]
    
    # Initialize row and column unique counts
    row_unique = [set() for _ in range(H)]
    col_unique = [set() for _ in range(W)]
    
    # Populate initial counts
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            row_counts[i][c] += 1
            col_counts[j][c] += 1
            row_unique[i].add(c)
            col_unique[j].add(c)
    
    while True:
        # Step 1: Mark rows
        marked_rows = set()
        for i in range(H):
            if row_size[i] >= 2 and len(row_unique[i]) == 1:
                marked_rows.add(i)
        
        # Step 2: Mark columns
        marked_cols = set()
        for j in range(W):
            if col_size[j] >= 2 and len(col_unique[j]) == 1:
                marked_cols.add(j)
        
        if not marked_rows and not marked_cols:
            break
        
        # Step 3: Remove marked cookies
        for i in marked_rows:
            for j in range(W):
                if remaining[i][j]:
                    c = grid[i][j]
                    col_counts[j][c] -= 1
                    if col_counts[j][c] == 0:
                        col_unique[j].discard(c)
                    remaining[i][j] = False
                    row_size[i] -= 1
                    col_size[j] -= 1
        
        for j in marked_cols:
            for i in range(H):
                if remaining[i][j]:
                    c = grid[i][j]
                    row_counts[i][c] -= 1
                    if row_counts[i][c] == 0:
                        row_unique[i].discard(c)
                    remaining[i][j] = False
                    row_size[i] -= 1
                    col_size[j] -= 1
    
    # Count remaining cookies
    count = 0
    for i in range(H):
        for j in range(W):
            if remaining[i][j]:
                count += 1
    print(count)

if __name__ == "__main__":
    main()