# Read the input
N = int(input())

# Read grid A
grid_a = []
for _ in range(N):
    row = list(input())
    grid_a.append(row)

# Read grid B
grid_b = []
for _ in range(N):
    row = list(input())
    grid_b.append(row)

# Find the different cell
for i in range(N):
    for j in range(N):
        if grid_a[i][j] != grid_b[i][j]:
            print(i+1, j+1)
            break