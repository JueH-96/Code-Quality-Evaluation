n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))

# Sort the cards based on their strength (A) in ascending order
cards_sorted = sorted(cards, key=lambda x: x[0])

min_c = float('inf')
result = []

# Iterate in reverse order (from highest A to lowest)
for card in reversed(cards_sorted):
    a, c, idx = card
    if c < min_c:
        result.append(idx)
        min_c = c

# Sort the result to ensure the output is in ascending order of indices
result.sort()

print(len(result))
print(' '.join(map(str, result)))