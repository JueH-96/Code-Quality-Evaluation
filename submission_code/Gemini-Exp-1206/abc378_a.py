colors = list(map(int, input().split()))
counts = {}
for color in colors:
    if color not in counts:
        counts[color] = 0
    counts[color] += 1
operations = 0
for count in counts.values():
    operations += count // 2
print(operations)