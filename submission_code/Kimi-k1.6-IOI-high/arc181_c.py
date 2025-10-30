n = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Compute inverse permutation for P
p_inv = [0] * (n + 1)
for idx in range(n):
    val = P[idx]
    p_inv[val] = idx + 1  # 1-based position

# Compute inverse permutation for Q
q_inv = [0] * (n + 1)
for idx in range(n):
    val = Q[idx]
    q_inv[val] = idx + 1  # 1-based position

# Compute column values
c = [0] * (n + 1)
for j in range(1, n + 1):
    c[j] = (n + 1) - q_inv[j]

# Generate the grid
grid = []
for i in range(1, n + 1):
    row = []
    for j in range(1, n + 1):
        if p_inv[i] < c[j]:
            row.append('0')
        else:
            row.append('1')
    grid.append(''.join(row))

# Print the grid
for line in grid:
    print(line)