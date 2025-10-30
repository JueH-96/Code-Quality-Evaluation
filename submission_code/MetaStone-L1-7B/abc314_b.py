n = int(input())
x = int(input())

candidates = []
for i in range(n):
    c_i = int(input())
    a_list = list(map(int, input().split()))
    if x in a_list:
        candidates.append((c_i, i + 1))

if not candidates:
    print(0)
else:
    min_c = min(c for c, i in candidates)
    selected = [i for c, i in candidates if c == min_c]
    selected.sort()
    print(len(selected))
    print(' '.join(map(str, selected)))