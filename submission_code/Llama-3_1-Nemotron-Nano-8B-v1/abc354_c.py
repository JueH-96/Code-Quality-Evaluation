n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((-a, c, i + 1))  # Use -a for descending sort

# Sort by descending A (so we use the negative A for sorting)
cards.sort()
min_c = float('inf')
result = []

for a, c, idx in cards:
    if c < min_c:
        result.append(idx)
        min_c = c

# Sort the result indices in ascending order
result.sort()
print(len(result))
print(' '.join(map(str, result)))