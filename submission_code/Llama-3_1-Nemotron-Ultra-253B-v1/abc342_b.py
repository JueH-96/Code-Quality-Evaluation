n = int(input())
p = list(map(int, input().split()))
pos = {person: i+1 for i, person in enumerate(p)}
q = int(input())
for _ in range(q):
    a, b = map(int, input().split())
    print(a if pos[a] < pos[b] else b)