n = int(input())
p = list(map(int, input().split()))
person_pos = {p[i]: i + 1 for i in range(n)}
q = int(input())
for _ in range(q):
    a, b = map(int, input().split())
    if person_pos[a] < person_pos[b]:
        print(a)
    else:
        print(b)