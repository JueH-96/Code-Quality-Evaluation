# Read input values
N, S, M, L = map(int, input().split())

# Calculate maximum possible counts for each pack
max_x = (N + 5) // 6
max_y = (N + 7) // 8
max_z = (N + 11) // 12

min_cost = float('inf')

# Iterate through all possible combinations
for x in range(max_x + 1):
    for y in range(max_y + 1):
        for z in range(max_z + 1):
            total_eggs = 6 * x + 8 * y + 12 * z
            if total_eggs >= N:
                current_cost = x * S + y * M + z * L
                if current_cost < min_cost:
                    min_cost = current_cost

# Output the minimum cost
print(min_cost)