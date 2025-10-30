import sys

def min_cost_to_buy_eggs(N, S, M, L):
    # Costs per egg for each pack
    cost_per_egg_6 = S / 6
    cost_per_egg_8 = M / 8
    cost_per_egg_12 = L / 12

    # Initialize a large number for minimum cost
    min_cost = float('inf')

    # Try all combinations of packs
    for pack6 in range((N // 6) + 1):
        for pack8 in range((N // 8) + 1):
            for pack12 in range((N // 12) + 1):
                total_eggs = 6 * pack6 + 8 * pack8 + 12 * pack12
                total_cost = S * pack6 + M * pack8 + L * pack12
                if total_eggs >= N:
                    min_cost = min(min_cost, total_cost)

    return min_cost

# Read input from stdin
input_line = sys.stdin.readline().strip().split()
N = int(input_line[0])
S = int(input_line[1])
M = int(input_line[2])
L = int(input_line[3])

# Calculate the minimum cost
result = min_cost_to_buy_eggs(N, S, M, L)

# Print the result to stdout
print(result)