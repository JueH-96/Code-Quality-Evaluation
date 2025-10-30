import sys

def process_queries(H, W, Q, queries):
    grid = [[True] * W for _ in range(H)]
    remaining_walls = H * W
    
    for r, c in queries:
        r -= 1
        c -= 1
        if grid[r][c]:
            grid[r][c] = False
            remaining_walls -= 1
        else:
            # Check up
            for i in range(r-1, -1, -1):
                if grid[i][c]:
                    grid[i][c] = False
                    remaining_walls -= 1
                    break
            # Check down
            for i in range(r+1, H):
                if grid[i][c]:
                    grid[i][c] = False
                    remaining_walls -= 1
                    break
            # Check left
            for j in range(c-1, -1, -1):
                if grid[r][j]:
                    grid[r][j] = False
                    remaining_walls -= 1
                    break
            # Check right
            for j in range(c+1, W):
                if grid[r][j]:
                    grid[r][j] = False
                    remaining_walls -= 1
                    break
    return remaining_walls

def main():
    input = sys.stdin.read
    data = input().split()
    
    H, W, Q = map(int, [data[0], data[1], data[2]])
    queries = [(int(data[i]), int(data[i+1])) for i in range(3, 2*Q+3, 2)]
    
    result = process_queries(H, W, Q, queries)
    print(result)

if __name__ == "__main__":
    main()