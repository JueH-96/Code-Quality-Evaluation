import sys
data = sys.stdin.read().split()
N, S, M, L = map(int, data)

min_cost = float('inf')

# Define upper bounds for a, b, c to ensure all possibilities are covered
a_upper = (N + 5) // 6 + 1  # Ceiling of N/6 plus one
b_upper = (N + 7) // 8 + 1  # Ceiling of N/8 plus one
c_upper = (N + 11) // 12 + 1  # Ceiling of N/12 plus one

# Iterate through all possible combinations of packs
for c in range(c_upper + 1):  # Number of 12-egg packs
    for b in range(b_upper + 1):  # Number of 8-egg packs
        for a in range(a_upper + 1):  # Number of 6-egg packs
            eggs = 6 * a + 8 * b + 12 * c
            if eggs >= N:
                cost = a * S + b * M + c * L
                if cost < min_cost:
                    min_cost = cost

# Output the minimum cost
print(min_cost)