# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Q = int(data[2])
    
    queries = []
    for i in range(Q):
        R = int(data[3 + 2 * i]) - 1
        C = int(data[4 + 2 * i]) - 1
        queries.append((R, C))
    
    # Initialize the grid with walls
    grid = [[True] * W for _ in range(H)]
    remaining_walls = H * W
    
    for R, C in queries:
        if grid[R][C]:
            grid[R][C] = False
            remaining_walls -= 1
        else:
            # Look up
            for i in range(R - 1, -1, -1):
                if grid[i][C]:
                    grid[i][C] = False
                    remaining_walls -= 1
                    break
            # Look down
            for i in range(R + 1, H):
                if grid[i][C]:
                    grid[i][C] = False
                    remaining_walls -= 1
                    break
            # Look left
            for j in range(C - 1, -1, -1):
                if grid[R][j]:
                    grid[R][j] = False
                    remaining_walls -= 1
                    break
            # Look right
            for j in range(C + 1, W):
                if grid[R][j]:
                    grid[R][j] = False
                    remaining_walls -= 1
                    break
    
    print(remaining_walls)

if __name__ == "__main__":
    main()