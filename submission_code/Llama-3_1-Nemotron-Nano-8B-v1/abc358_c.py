import itertools

n, m = map(int, input().split())
sets = []
full = set(range(1, m + 1))

for _ in range(n):
    s = input().strip()
    flavors = set()
    for j in range(m):
        if s[j] == 'o':
            flavors.add(j + 1)
    sets.append(flavors)

for k in range(1, n + 1):
    for combo in itertools.combinations(sets, k):
        union = set()
        for s in combo:
            union.update(s)
        if union == full:
            print(k)
            exit()