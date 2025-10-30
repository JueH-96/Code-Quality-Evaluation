# Read input
H, W, Y = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(map(int, input().split())))

# Function to check if a cell is sinkable
def is_sinkable(i, j, sea_level):
    if i < 0 or i >= H or j < 0 or j >= W:
        return False
    return grid[i][j] <= sea_level

# Sink cells and return the remaining area
def sink_cells(i, j, sea_level):
    if not is_sinkable(i, j, sea_level):
        return 1
    grid[i][j] = 0
    return (
        1
        + sink_cells(i-1, j, sea_level)
        + sink_cells(i+1, j, sea_level)
        + sink_cells(i, j-1, sea_level)
        + sink_cells(i, j+1, sea_level)
    )

# Solve the problem
for year in range(1, Y+1):
    remaining_area = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] > year:
                remaining_area += sink_cells(i, j, year)
    print(remaining_area)