# YOUR CODE HERE
N, S, M, L = map(int, input().split())

min_cost = float('inf')

# Try all combinations of pack counts
# We don't need more than N eggs + max pack size to ensure we check all possibilities
max_packs = N // 6 + 2  # A reasonable upper bound

for pack6 in range(max_packs):
    for pack8 in range(max_packs):
        for pack12 in range(max_packs):
            total_eggs = 6 * pack6 + 8 * pack8 + 12 * pack12
            if total_eggs >= N:
                cost = S * pack6 + M * pack8 + L * pack12
                min_cost = min(min_cost, cost)

print(min_cost)