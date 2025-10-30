n = int(input())
a = list(map(int, input().split()))
pos = [[] for _ in range(n + 1)]  # Using 1-based index for colors

for idx in range(len(a)):
    color = a[idx]
    pos[color].append(idx + 1)  # Positions are 1-based

count = 0
for color in range(1, n + 1):
    p1, p2 = pos[color]
    if p2 - p1 == 2:
        count += 1

print(count)