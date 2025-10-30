n = int(input())
people = []
for _ in range(n):
    c = int(input())
    a_list = list(map(int, input().split()))
    people.append((c, a_list))
x = int(input())

candidates = []
for i in range(n):
    person_num = i + 1
    c_i, a_list = people[i]
    if x in a_list:
        candidates.append((person_num, c_i))

if not candidates:
    print(0)
else:
    min_c = min(c[1] for c in candidates)
    selected = [person for person, c in candidates if c == min_c]
    selected.sort()
    print(len(selected))
    print(' '.join(map(str, selected)))