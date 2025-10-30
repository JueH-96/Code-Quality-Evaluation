# YOUR CODE HERE
def main():
    import sys
    H, W, N = map(int, sys.stdin.read().split())
    grid = [[False for _ in range(W)] for _ in range(H)]
    # Directions: 0-up,1-right,2-down,3-left
    dr = [-1,0,1,0]
    dc = [0,1,0,-1]
    direction = 0
    r, c = 0, 0
    for _ in range(N):
        if not grid[r][c]:
            grid[r][c] = True
            direction = (direction +1) %4
        else:
            grid[r][c] = False
            direction = (direction -1) %4
        r = (r + dr[direction]) % H
        c = (c + dc[direction]) % W
    for row in grid:
        print(''.join(['#' if cell else '.' for cell in row]))

if __name__ == "__main__":
    main()