n = int(input())
cards = []
for idx in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, idx + 1))

# Sort the cards by their strength (A)
cards.sort()

min_c = float('inf')
min_c_suffix = [0] * n

# Compute the minimum cost suffix array
for i in reversed(range(n)):
    min_c_suffix[i] = min_c
    if cards[i][1] < min_c:
        min_c = cards[i][1]

kept = []
for i in range(n):
    if min_c_suffix[i] >= cards[i][1]:
        kept.append(cards[i][2])

# Sort the kept indices and print
kept.sort()
print(len(kept))
print(' '.join(map(str, kept)))