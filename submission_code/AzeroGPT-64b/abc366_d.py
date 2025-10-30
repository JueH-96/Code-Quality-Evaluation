import numpy as np

n, = map(int, input().split())
ar = np.zeros((n + 1, n + 1, n + 1), dtype=np.int64)
for y in range(1, n + 1):
    for line in range(n):
        tmp = np.fromstring(input(), sep=' ', dtype=np.int64)
        ar[y, line + 1] = tmp

for x in range(1, n + 1):
    for y in range(1, n + 1):
        ar[x, y] = np.cumsum(ar[x, y])

for y in range(1, n + 1):
    ar[:, y] = np.cumsum(ar[:, y], axis=0)

ar = np.cumsum(ar, axis=2)

q, = map(int, input().split())
for _ in range(q):
    lx, rx, ly, ry, lz, rz = (int(x) for x in input().split())
    res = 0
    res += ar[rx, ry, rz] - ar[rx, ry, lz - 1]
    res += ar[rx, ly - 1, rz] - ar[rx, ly - 1, lz - 1]
    res += ar[lx - 1, ry, rz] - ar[lx - 1, ry, lz - 1]
    res += ar[lx - 1, ly - 1, rz] - ar[lx - 1, ly - 1, lz - 1]
    res -= ar[lx - 1, ry, rz] - ar[lx - 1, ry, lz - 1]
    res -= ar[rx, ly - 1, rz] - ar[rx, ly - 1, lz - 1]
    res -= ar[lx - 1, ry, rz] - ar[lx - 1, ry, lz - 1]
    res -= ar[rx, ly - 1, rz] - ar[rx, ly - 1, lz - 1]
    print(res)