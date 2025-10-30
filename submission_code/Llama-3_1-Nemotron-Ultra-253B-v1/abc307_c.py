# Read input
def read_grid():
    h, w = map(int, input().split())
    grid = [input().strip() for _ in range(h)]
    return grid

A = read_grid()
B = read_grid()
X = read_grid()

# Collect black squares for A and B
HA, WA = len(A), len(A[0]) if HA > 0 else 0
HB, WB = len(B), len(B[0]) if HB > 0 else 0
HX, WX = len(X), len(X[0]) if HX > 0 else 0

black_A = [(i, j) for i in range(HA) for j in range(WA) if A[i][j] == '#']
black_B = [(i, j) for i in range(HB) for j in range(WB) if B[i][j] == '#']

# Function to compute ranges for a sheet's positions
def get_ranges(black, H, W, HX, WX):
    if not black:
        return None, None, None, None  # Should not happen per problem statement
    xs = [i for (i, j) in black]
    ys = [j for (i, j) in black]
    a_x_min = max(-i for i in xs)
    a_x_max = min(HX - i - 1 for i in xs)
    a_y_min = max(-j for j in ys)
    a_y_max = min(WX - j - 1 for j in ys)
    return a_x_min, a_x_max, a_y_min, a_y_max

# Get ranges for A and B
a_x_min, a_x_max, a_y_min, a_y_max = get_ranges(black_A, HA, WA, HX, WX)
b_x_min, b_x_max, b_y_min, b_y_max = get_ranges(black_B, HB, WB, HX, WX)

# Check if ranges are valid
if a_x_min > a_x_max or a_y_min > a_y_max or b_x_min > b_x_max or b_y_min > b_y_max:
    print("No")
    exit()

# Iterate all possible positions
for a_x in range(a_x_min, a_x_max + 1):
    for a_y in range(a_y_min, a_y_max + 1):
        for b_x in range(b_x_min, b_x_max + 1):
            for b_y in range(b_y_min, b_y_max + 1):
                # Create the grid
                grid = [['.' for _ in range(WX)] for _ in range(HX)]
                # Place A's blacks
                for i, j in black_A:
                    x = a_x + i
                    y = a_y + j
                    grid[x][y] = '#'
                # Place B's blacks
                for i, j in black_B:
                    x = b_x + i
                    y = b_y + j
                    grid[x][y] = '#'
                # Check if matches X
                match = True
                for x_row in range(HX):
                    row = ''.join(grid[x_row])
                    if row != X[x_row]:
                        match = False
                        break
                if match:
                    print("Yes")
                    exit()

print("No")