# YOUR CODE HERE
def solve():
    h, w, q = map(int, input().split())
    grid = [[1] * w for _ in range(h)]
    
    for _ in range(q):
        r, c = map(int, input().split())
        r -= 1
        c -= 1
        
        if grid[r][c] == 1:
            grid[r][c] = 0
        else:
            # Up
            for i in range(r - 1, -1, -1):
                if grid[i][c] == 1:
                    grid[i][c] = 0
                    break
            # Down
            for i in range(r + 1, h):
                if grid[i][c] == 1:
                    grid[i][c] = 0
                    break
            # Left
            for j in range(c - 1, -1, -1):
                if grid[r][j] == 1:
                    grid[r][j] = 0
                    break
            # Right
            for j in range(c + 1, w):
                if grid[r][j] == 1:
                    grid[r][j] = 0
                    break
                    
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 1:
                count += 1
    print(count)

solve()