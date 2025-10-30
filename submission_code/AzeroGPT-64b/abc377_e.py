import sys
input = sys.stdin.buffer.readline

def find(x, i):
    if i[x] < 0:
        return x
    else:
        i[x] = find(i[i[x]], i)
        return i[x]

def unite(x, y, k, i):
    x = find(x, i)
    y = find(y, i)
    dif = k + 1
    if x == y:
        return
    if i[x] > i[y]:
        x, y = y, x
    i[x] += i[y]
    i[y] = x
    minimum[x] = min(minimum[x], dif, minimum[y])
    maximum[x] = max(maximum[x], dif, maximum[y])

def compress(x, cy, K, i):
    root = find(x, i)
    if root == x and minimum[root] * 2 > K:
        return cy
    else:
        return compress(x, cy, maximum[root] + (K - maximum[root]) % (maximum[root] - minimum[root] + 1), i) #mod

N, K = map(int, input().split())
P = list(map(int, input().split()))
i = [-1] * (N + 1)
minimum = [float('inf')] * (N + 1)
maximum = [0] * (N + 1)
for x, y in enumerate(P, 1):
    unite(x, y, K, i)
for x in range(1, N + 1):
    print(compress(x, P[x - 1], K, i), end=" ")