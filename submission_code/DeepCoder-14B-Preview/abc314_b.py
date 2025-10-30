n = int(input())

people = []
for i in range(n):
    c = int(input())
    a = list(map(int, input().split()))
    bets = set(a)
    people.append((c, bets, i + 1))

x = int(input())

candidates = []
for person in people:
    if x in person[1]:
        candidates.append((person[0], person[2]))

if not candidates:
    print(0)
else:
    min_c = min(c for c, num in candidates)
    selected = [num for c, num in candidates if c == min_c]
    selected.sort()
    print(len(selected))
    if selected:
        print(' '.join(map(str, selected)))