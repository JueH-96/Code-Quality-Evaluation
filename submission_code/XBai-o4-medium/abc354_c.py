n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))

# Sort cards by descending strength (A), then ascending cost (C) [though not necessary, just for consistency]
sorted_cards = sorted(cards, key=lambda x: (-x[0], x[1]))

min_c = 10**18  # A large enough value to start with
result = []
for card in sorted_cards:
    a, c, idx = card
    if c < min_c:
        result.append(idx)
        min_c = c

result.sort()
print(len(result))
print(' '.join(map(str, result)))