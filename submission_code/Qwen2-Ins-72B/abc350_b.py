N, Q = map(int, input().split())
treatments = list(map(int, input().split()))

teeth = set(range(1, N+1))

for treatment in treatments:
    if treatment in teeth:
        teeth.remove(treatment)
    else:
        teeth.add(treatment)

print(len(teeth))