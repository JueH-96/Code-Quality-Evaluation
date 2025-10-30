def min_cost_to_buy_eggs(N, S, M, L):
    # Initialize a large number for comparison
    min_cost = float('inf')
    
    # Iterate through the number of packs of 6 eggs
    for packs_of_6 in range((N // 6) + 2):  # +2 to cover the case where we need one more pack
        for packs_of_8 in range((N // 8) + 2):
            for packs_of_12 in range((N // 12) + 2):
                total_eggs = (packs_of_6 * 6) + (packs_of_8 * 8) + (packs_of_12 * 12)
                if total_eggs >= N:
                    cost = (packs_of_6 * S) + (packs_of_8 * M) + (packs_of_12 * L)
                    min_cost = min(min_cost, cost)
    
    return min_cost

import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
S = int(data[1])
M = int(data[2])
L = int(data[3])

result = min_cost_to_buy_eggs(N, S, M, L)
print(result)