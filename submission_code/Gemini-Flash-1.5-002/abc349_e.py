grid = []
for _ in range(3):
    grid.append(list(map(int, input().split())))

total_sum = sum(sum(row) for row in grid)

if total_sum > 0:
    print("Takahashi")
else:
    print("Aoki")