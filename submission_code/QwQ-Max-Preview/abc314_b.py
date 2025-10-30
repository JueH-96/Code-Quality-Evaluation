n = int(input())
people = []
for _ in range(n):
    c = int(input())
    a = list(map(int, input().split()))
    people.append((c, a))
x = int(input())

candidates = []
for idx in range(n):
    c_i, bets = people[idx]
    if x in bets:
        candidates.append((c_i, idx + 1))  # Using 1-based index

if not candidates:
    print(0)
else:
    min_c = min(c[0] for c in candidates)
    selected = [person_num for c_i, person_num in candidates if c_i == min_c]
    selected.sort()
    print(len(selected))
    print(' '.join(map(str, selected)))