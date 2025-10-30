import sys

# Read all input from stdin
data = sys.stdin.read().split()
idx = 0

# Read N
N = int(data[idx])
idx += 1

# Read cards with their original indices
cards = []
for card_num in range(1, N + 1):
    A = int(data[idx])
    C = int(data[idx + 1])
    idx += 2
    cards.append((A, C, card_num))

# Sort cards by strength A in ascending order
sorted_cards = sorted(cards, key=lambda x: x[0])

# Extract cost list from sorted cards
cost_list = [card[1] for card in sorted_cards]

# Compute suffix minimum costs
suffix_min = [0] * N
suffix_min[N - 1] = cost_list[N - 1]
for i in range(N - 2, -1, -1):
    suffix_min[i] = min(cost_list[i], suffix_min[i + 1])

# Define a large number for infinity
INF = 1000000001

# Find indices of non-dominated cards
remaining_indices = []
for i in range(N):
    if i < N - 1:
        min_after_val = suffix_min[i + 1]
    else:
        min_after_val = INF
    if cost_list[i] <= min_after_val:
        idx_orig = sorted_cards[i][2]
        remaining_indices.append(idx_orig)

# Sort the remaining indices in ascending order
remaining_indices.sort()

# Output the result
m = len(remaining_indices)
print(m)
print(' '.join(map(str, remaining_indices)))