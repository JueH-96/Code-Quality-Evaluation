n = int(input())
people = []
for _ in range(n):
    c = int(input())
    a = set(map(int, input().split()))
    people.append((c, a))
x = int(input())

candidates = []
for i in range(n):
    ci, ai = people[i]
    if x in ai:
        candidates.append((i + 1, ci))  # Person numbers are 1-based

if not candidates:
    print(0)
else:
    min_c = min(c[1] for c in candidates)
    selected = [c[0] for c in candidates if c[1] == min_c]
    selected.sort()
    print(len(selected))
    print(' '.join(map(str, selected)))