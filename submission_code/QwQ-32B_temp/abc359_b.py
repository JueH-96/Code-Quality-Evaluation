n = int(input())
a = list(map(int, input().split()))
positions = {}

for idx in range(len(a)):
    color = a[idx]
    pos = idx + 1  # Convert to 1-based position
    if color not in positions:
        positions[color] = [pos]
    else:
        positions[color].append(pos)

count = 0
for i in range(1, n+1):
    first, second = positions[i]
    if second - first == 2:
        count += 1

print(count)