n = int(input())
p = list(map(int, input().split()))
positions = [0] * (n + 1)
for i in range(n):
    positions[p[i]] = i + 1
q = int(input())
for _ in range(q):
    a, b = map(int, input().split())
    if positions[a] < positions[b]:
        print(a)
    else:
        print(b)