import itertools

n, m = map(int, input().split())
stands = []
for _ in range(n):
    s = input().strip()
    flavor = set()
    for j in range(m):
        if s[j] == 'o':
            flavor.add(j + 1)
    stands.append(flavor)

for k in range(1, n + 1):
    for combo in itertools.combinations(stands, k):
        union = set()
        for st in combo:
            union.update(st)
        if len(union) == m:
            print(k)
            exit()