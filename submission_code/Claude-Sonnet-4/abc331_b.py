N, S, M, L = map(int, input().split())

min_cost = float('inf')

# Try all combinations of packs
# We don't need more than N/6 + some buffer for 6-egg packs
# We don't need more than N/8 + some buffer for 8-egg packs  
# We don't need more than N/12 + some buffer for 12-egg packs
# Since N <= 100, we can safely limit to 20 of each type

for pack6 in range(21):  # 0 to 20 packs of 6 eggs
    for pack8 in range(21):  # 0 to 20 packs of 8 eggs
        for pack12 in range(21):  # 0 to 20 packs of 12 eggs
            total_eggs = pack6 * 6 + pack8 * 8 + pack12 * 12
            if total_eggs >= N:
                cost = pack6 * S + pack8 * M + pack12 * L
                min_cost = min(min_cost, cost)

print(min_cost)