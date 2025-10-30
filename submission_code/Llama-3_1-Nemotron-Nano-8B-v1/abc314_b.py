n = int(input())
people = []
for _ in range(n):
    c = int(input())
    a = list(map(int, input().split()))
    people.append((c, set(a)))
x = int(input())

candidates = []
for idx in range(n):
    c, a_set = people[idx]
    if x in a_set:
        candidates.append((c, idx + 1))

if not candidates:
    print(0)
else:
    min_c = min(c for c, _ in candidates)
    selected = [num for c, num in candidates if c == min_c]
    selected.sort()
    print(len(selected))
    print(' '.join(map(str, selected)))