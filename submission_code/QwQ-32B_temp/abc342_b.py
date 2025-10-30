n = int(input())
p = list(map(int, input().split()))
pos = {}
for idx in range(n):
    pos[p[idx]] = idx + 1
q = int(input())
for _ in range(q):
    a, b = map(int, input().split())
    if pos[a] < pos[b]:
        print(a)
    else:
        print(b)