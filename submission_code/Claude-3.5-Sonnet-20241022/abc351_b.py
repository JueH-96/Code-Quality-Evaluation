N = int(input())
grid_a = []
grid_b = []

# Read grid A
for _ in range(N):
    grid_a.append(input())

# Read grid B 
for _ in range(N):
    grid_b.append(input())

# Find the differing cell
for i in range(N):
    for j in range(N):
        if grid_a[i][j] != grid_b[i][j]:
            print(i+1, j+1)
            break