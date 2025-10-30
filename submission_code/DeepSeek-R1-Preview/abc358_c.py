import itertools

n, m = map(int, input().split())
flavors = []
for _ in range(n):
    s = input().strip()
    current = set()
    for j in range(m):
        if s[j] == 'o':
            current.add(j)
    flavors.append(current)

for k in range(1, n + 1):
    for combo in itertools.combinations(range(n), k):
        union = set()
        for i in combo:
            union.update(flavors[i])
        if len(union) == m:
            print(k)
            exit()

print(n)