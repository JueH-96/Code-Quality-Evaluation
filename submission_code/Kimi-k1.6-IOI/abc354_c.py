n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))

# Sort the cards in descending order of strength (A)
cards.sort(reverse=True, key=lambda x: x[0])

min_cost = float('inf')
kept = []

for a, c, idx in cards:
    if c < min_cost:
        kept.append(idx)
        min_cost = c

# Sort the kept indices in ascending order
kept.sort()

print(len(kept))
print(' '.join(map(str, kept)))