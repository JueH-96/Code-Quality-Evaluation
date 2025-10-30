n = int(input())
p = list(map(int, input().split()))
pos = {person: i + 1 for i, person in enumerate(p)}
q = int(input())
for _ in range(q):
    a, b = map(int, input().split())
    if pos[a] < pos[b]:
        print(a)
    else:
        print(b)