n, q = map(int, input().split())
treatments = list(map(int, input().split()))

teeth = set(range(1, n + 1))

for t in treatments:
    if t in teeth:
        teeth.remove(t)
    else:
        teeth.add(t)

print(len(teeth))