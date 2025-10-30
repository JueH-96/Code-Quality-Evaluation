n, k = map(int, input().split())
a = list(map(int, input().split()))
empty = k
count = 0
groups = a.copy()

while groups:
    if empty < groups[0]:
        count += 1
        empty = k
        empty -= groups[0]
        groups.pop(0)
    else:
        empty -= groups[0]
        groups.pop(0)

print(count)