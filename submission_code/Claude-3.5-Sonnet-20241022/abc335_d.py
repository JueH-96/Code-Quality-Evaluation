def solve():
    N = int(input())
    
    # Create empty grid
    grid = [[0]*N for _ in range(N)]
    
    # Place Takahashi at center
    center = N//2
    grid[center][center] = 'T'
    
    # Helper to check if position is valid
    def is_valid(x, y):
        return 0 <= x < N and 0 <= y < N and grid[x][y] == 0
    
    # Start from top-left corner and spiral clockwise
    dx = [0, 1, 0, -1]  # right, down, left, up
    dy = [1, 0, -1, 0]
    
    x, y = 0, 0
    curr_dir = 0
    num = 1
    
    # Place numbers 1 to N^2-1 in spiral
    while num <= N*N-1:
        if (x != center or y != center):  # Skip center position
            grid[x][y] = num
            num += 1
            
        # Try to go in current direction
        nx = x + dx[curr_dir]
        ny = y + dy[curr_dir]
        
        # If can't go in current direction, change direction
        if not is_valid(nx, ny):
            curr_dir = (curr_dir + 1) % 4
            nx = x + dx[curr_dir]
            ny = y + dy[curr_dir]
            
        x, y = nx, ny
    
    # Print result
    for row in grid:
        print(*row)

solve()