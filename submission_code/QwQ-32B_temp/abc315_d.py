import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())
    
    present_row = [True] * H
    present_col = [True] * W
    
    while True:
        eligible_rows = []
        eligible_cols = []
        
        # Check eligible rows
        for r in range(H):
            if not present_row[r]:
                continue
            first = None
            all_same = True
            count = 0
            for c in range(W):
                if present_col[c]:
                    count += 1
                    if first is None:
                        first = grid[r][c]
                    elif grid[r][c] != first:
                        all_same = False
                        break
            if all_same and count >= 2:
                eligible_rows.append(r)
        
        # Check eligible columns
        for c in range(W):
            if not present_col[c]:
                continue
            first = None
            all_same = True
            count = 0
            for r in range(H):
                if present_row[r]:
                    count += 1
                    if first is None:
                        first = grid[r][c]
                    elif grid[r][c] != first:
                        all_same = False
                        break
            if all_same and count >= 2:
                eligible_cols.append(c)
        
        if not eligible_rows and not eligible_cols:
            break
        
        # Remove eligible rows and columns
        for r in eligible_rows:
            present_row[r] = False
        for c in eligible_cols:
            present_col[c] = False
    
    # Count remaining cookies
    count = 0
    for r in range(H):
        if present_row[r]:
            for c in range(W):
                if present_col[c]:
                    count += 1
    print(count)

if __name__ == "__main__":
    main()