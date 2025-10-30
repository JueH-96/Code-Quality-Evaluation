n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))

# Sort cards in decreasing order of A
cards_sorted = sorted(cards, key=lambda x: -x[0])

min_c = float('inf')
result = []
for card in cards_sorted:
    a, c, idx = card
    if c < min_c:
        result.append(idx)
        min_c = c

# Sort the result in increasing order of indices
result.sort()
print(len(result))
print(' '.join(map(str, result)))