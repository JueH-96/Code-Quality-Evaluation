from collections import deque

N, Q = map(int, input().split())
dragon = deque([(i, 0) for i in range(1, N+1)])

for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        direction = query[1]
        if direction == 'R':
            dragon.appendleft((dragon[0][0]+1, dragon[0][1]))
            dragon.pop()
        elif direction == 'L':
            dragon.append((dragon[-1][0]-1, dragon[-1][1]))
            dragon.popleft()
        elif direction == 'U':
            dragon.appendleft((dragon[0][0], dragon[0][1]+1))
            dragon.pop()
        else:
            dragon.append((dragon[-1][0], dragon[-1][1]-1))
            dragon.popleft()
    else:
        p = int(query[1])
        print(f"{dragon[p-1][0]} {dragon[p-1][1]}")