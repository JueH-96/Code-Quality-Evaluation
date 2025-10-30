n = int(input())
a = list(map(int, input().split()))
pos = [[] for _ in range(n + 1)]
for idx in range(len(a)):
    num = a[idx]
    pos[num].append(idx + 1)
res = []
for i in range(1, n + 1):
    res.append((pos[i][1], i))
res.sort()
output = [str(x[1]) for x in res]
print(' '.join(output))