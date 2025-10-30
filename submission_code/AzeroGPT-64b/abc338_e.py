import sys

n = int(input())
edges = []
for _ in range(n):
    a, b = map(int, input().split())
    if a > b: a, b = b, a
    edges.append((a, b))
s = sorted(edges)
t = sorted(edges, key=lambda x: x[1])
i = 0
for j in range(n):
    if t[j][0] > s[i][1]:
        print('Yes')
        sys.exit()
    if t[j][1] == s[i][1]:
        i += 1
print('No')