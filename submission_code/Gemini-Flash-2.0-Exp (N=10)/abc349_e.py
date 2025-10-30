def solve():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, input().split())))

    if grid[1][1] != 0:
        print("Takahashi")
    else:
        print("Aoki")

solve()