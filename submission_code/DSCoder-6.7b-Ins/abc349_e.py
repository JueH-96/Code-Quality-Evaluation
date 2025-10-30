# YOUR CODE HERE

def read_grid():
    grid = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    return grid

def check_win(grid):
    # Check rows
    for row in grid:
        if sum(row) % 2 == 0:
            return False
    # Check columns
    for i in range(3):
        if sum(grid[j][i] for j in range(3)) % 2 == 0:
            return False
    # Check diagonals
    if sum(grid[i][i] for i in range(3)) % 2 == 0 or sum(grid[i][2 - i] for i in range(3)) % 2 == 0:
        return False
    return True

def main():
    grid = read_grid()
    if check_win(grid):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()