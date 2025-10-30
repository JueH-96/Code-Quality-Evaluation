n = int(input())
cards = []
for idx in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, idx + 1))

# Sort the cards in descending order of strength (A)
cards_sorted = sorted(cards, key=lambda x: -x[0])

min_cost = float('inf')
remaining = []

for a, c, idx in cards_sorted:
    if c < min_cost:
        remaining.append(idx)
        min_cost = c

# Sort the indices in ascending order
remaining.sort()

print(len(remaining))
print(' '.join(map(str, remaining)))