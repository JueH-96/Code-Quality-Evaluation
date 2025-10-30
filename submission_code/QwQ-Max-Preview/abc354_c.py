n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))

# Sort the cards in descending order of strength (A)
cards.sort(key=lambda x: -x[0])

min_cost = float('inf')
result = []

for a, c, idx in cards:
    if c < min_cost:
        result.append(idx)
        min_cost = c

# Sort the result indices in ascending order
result.sort()

print(len(result))
print(' '.join(map(str, result)))