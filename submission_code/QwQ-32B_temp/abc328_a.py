n, x = map(int, input().split())
scores = list(map(int, input().split()))
total = 0
for s in scores:
    if s <= x:
        total += s
print(total)