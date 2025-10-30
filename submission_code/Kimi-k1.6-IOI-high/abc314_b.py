n = int(input())
people = []
for _ in range(n):
    c = int(input())
    a = list(map(int, input().split()))
    s = set(a)
    people.append((c, s))
x = int(input())

candidates = []
for i in range(n):
    c_i, s = people[i]
    if x in s:
        candidates.append((i + 1, c_i))  # 1-based index

if not candidates:
    print(0)
    print()
else:
    min_c = min(c for (idx, c) in candidates)
    result = [idx for (idx, c) in candidates if c == min_c]
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))