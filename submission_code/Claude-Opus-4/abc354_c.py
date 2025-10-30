# YOUR CODE HERE
n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))  # (strength, cost, 1-indexed position)

# Sort by strength in descending order
cards.sort(key=lambda x: -x[0])

remaining = []
min_cost = float('inf')

for strength, cost, idx in cards:
    if cost < min_cost:
        # This card is not dominated by any stronger card
        remaining.append(idx)
        min_cost = cost

# Sort by index for output
remaining.sort()

print(len(remaining))
print(*remaining)