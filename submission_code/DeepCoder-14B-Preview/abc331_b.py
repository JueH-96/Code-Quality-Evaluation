# Read input values
N, S, M, L = map(int, input().split())

min_cost = float('inf')

# Calculate the maximum possible number of each pack to consider
max_a = (N // 6) + 4
max_b = (N // 8) + 4
max_c = (N // 12) + 4

# Iterate through all possible combinations of a, b, c
for a in range(max_a):
    for b in range(max_b):
        for c in range(max_c):
            total_eggs = 6 * a + 8 * b + 12 * c
            if total_eggs >= N:
                current_cost = a * S + b * M + c * L
                if current_cost < min_cost:
                    min_cost = current_cost

# Output the minimum cost found
print(min_cost)