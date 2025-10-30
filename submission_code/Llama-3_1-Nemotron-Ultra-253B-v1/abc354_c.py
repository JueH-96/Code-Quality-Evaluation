n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((-a, c, i + 1))

cards.sort()
min_c = float('inf')
result = []

for a_neg, c, idx in cards:
    if c < min_c:
        result.append(idx)
        min_c = c

result.sort()
print(len(result))
print(' '.join(map(str, result)))