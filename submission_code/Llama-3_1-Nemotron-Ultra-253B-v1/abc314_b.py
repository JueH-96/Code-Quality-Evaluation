n = int(input())
people = []
for i in range(n):
    c_i = int(input())
    a_list = list(map(int, input().split()))
    a_set = set(a_list)
    people.append((c_i, a_set, i + 1))

x = int(input())

selected = []
for c, a_set, idx in people:
    if x in a_set:
        selected.append((c, idx))

if not selected:
    print(0)
    print()
else:
    min_c = min(c for c, idx in selected)
    result = [idx for c, idx in selected if c == min_c]
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))