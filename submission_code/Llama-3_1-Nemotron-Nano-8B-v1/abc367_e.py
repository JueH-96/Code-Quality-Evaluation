n, k = map(int, input().split())
x = list(map(int, input().split()))
a = list(map(int, input().split()))

if k == 0:
    print(' '.join(map(str, a)))
    exit()

max_level = 60
up = [[0] * (n + 1) for _ in range(max_level)]

for i in range(1, n + 1):
    up[0][i] = x[i - 1]

for m in range(1, max_level):
    for i in range(1, n + 1):
        up[m][i] = up[m - 1][up[m - 1][i]]

result = [0] * n
for i in range(1, n + 1):
    current = i
    remaining = k
    for m in reversed(range(max_level)):
        if remaining >= (1 << m):
            current = up[m][current]
            remaining -= (1 << m)
    result[i - 1] = a[current - 1]

print(' '.join(map(str, result)))