# YOUR CODE HERE
grid = [list(map(int, input().split())) for _ in range(3)]
total = sum(sum(row) for row in grid)
if total > 0:
    print("Takahashi")
else:
    print("Aoki")