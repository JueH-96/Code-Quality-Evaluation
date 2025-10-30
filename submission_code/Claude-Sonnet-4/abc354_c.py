n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))  # (strength, cost, original_index)

# Sort by strength in descending order
cards.sort(key=lambda x: x[0], reverse=True)

remaining = []
min_cost = float('inf')

for strength, cost, idx in cards:
    if cost < min_cost:
        remaining.append(idx)
        min_cost = cost

# Sort remaining cards by their original indices
remaining.sort()

print(len(remaining))
print(*remaining)