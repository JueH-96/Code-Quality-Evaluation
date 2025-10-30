from collections import deque

n, q = map(int, input().split())
parts = deque([(i, 0) for i in range(n)])

ans = []
for _ in range(q):
    cmd, x = input().split()
    x = int(x)

    if cmd == "1":
        delta = {"R": (1, 0), "L": (-1, 0), "U": (0, 1), "D": (0, -1)}[x]
        head_new = (parts[0][0]+delta[0], parts[0][1]+delta[1])
        parts.appendleft(head_new)
        parts.pop()
    else:
        ans.append(parts[x-1])

for a, b in ans:
    print(a, b)