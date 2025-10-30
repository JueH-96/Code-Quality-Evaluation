n = int(input())
people = []
for _ in range(n):
    c = int(input())
    a_list = list(map(int, input().split()))
    people.append((c, a_list))

x = int(input())

candidates = []
for idx in range(n):
    person_num = idx + 1
    c_i, bets = people[idx]
    if x in bets:
        candidates.append((person_num, c_i))

if not candidates:
    print(0)
else:
    min_c = min(c for (i, c) in candidates)
    selected = [i for (i, c) in candidates if c == min_c]
    selected.sort()
    print(len(selected))
    print(' '.join(map(str, selected)))