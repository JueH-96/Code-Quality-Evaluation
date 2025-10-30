n = int(input())
p = list(map(int, input().split()))
q = list(map(int, input().split()))

# Compute row ranks (r) and column ranks (c)
r = [0] * (n + 1)
for idx in range(n):
    val = p[idx]
    r[val] = idx + 1

c = [0] * (n + 1)
for idx in range(n):
    val = q[idx]
    c[val] = idx + 1

# Generate the grid
grid = []
for i in range(1, n + 1):
    row = []
    for j in range(1, n + 1):
        effective_rank_j = n - c[j] + 1
        if r[i] < effective_rank_j:
            row.append('0')
        else:
            row.append('1')
    grid.append(''.join(row))

# Print the grid
for line in grid:
    print(line)