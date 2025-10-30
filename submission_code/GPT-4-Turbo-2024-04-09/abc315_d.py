def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [list(data[i + 2]) for i in range(H)]
    
    row_marked = [False] * H
    col_marked = [False] * W
    
    while True:
        any_marked = False
        
        # Check rows
        for i in range(H):
            if not row_marked[i]:
                same_color = True
                first_color = grid[i][0]
                for j in range(W):
                    if grid[i][j] != first_color:
                        same_color = False
                        break
                if same_color:
                    row_marked[i] = True
                    any_marked = True
        
        # Check columns
        for j in range(W):
            if not col_marked[j]:
                same_color = True
                first_color = grid[0][j]
                for i in range(H):
                    if grid[i][j] != first_color:
                        same_color = False
                        break
                if same_color:
                    col_marked[j] = True
                    any_marked = True
        
        if not any_marked:
            break
        
        # Remove marked rows and columns
        for i in range(H):
            if row_marked[i]:
                for j in range(W):
                    grid[i][j] = '.'
        
        for j in range(W):
            if col_marked[j]:
                for i in range(H):
                    grid[i][j] = '.'
    
    # Count remaining cookies
    remaining_cookies = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '.':
                remaining_cookies += 1
    
    print(remaining_cookies)

if __name__ == "__main__":
    main()