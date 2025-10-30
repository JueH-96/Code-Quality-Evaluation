from collections import deque

N, Q = map(int, input().split())
dragon = deque([(i, 0) for i in range(N)])

ans = []
for _ in range(Q):
    query = list(map(str, input().split()))
    if query[0] == "1":
        if query[1] == "R":
            dragon.append(dragon[-1])
            dragon[-2] = dragon[-1]
            dragon[-1] = (dragon[-1][0] + 1, dragon[-1][1])
        elif query[1] == "L":
            dragon.appendleft(dragon[0])
            dragon[1] = dragon[0]
            dragon[0] = (dragon[0][0] - 1, dragon[0][1])
        elif query[1] == "U":
            dragon.append(dragon[-1])
            dragon[-2] = dragon[-1]
            dragon[-1] = (dragon[-1][0], dragon[-1][1] + 1)
        else:
            dragon.appendleft(dragon[0])
            dragon[1] = dragon[0]
            dragon[0] = (dragon[0][0], dragon[0][1] - 1)
    else:
        ans.append(dragon[int(query[1]) - 1])

for a in ans:
    print(a[0], a[1])