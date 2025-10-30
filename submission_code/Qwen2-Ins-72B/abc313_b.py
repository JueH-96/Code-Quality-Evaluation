from collections import deque

N, M = map(int, input().split())
wins = [[] for _ in range(N)]
loses = [0] * N
for _ in range(M):
    a, b = map(int, input().split())
    wins[a-1].append(b-1)
    loses[b-1] += 1

q = deque([i for i in range(N) if loses[i] == 0])
while q:
    x = q.popleft()
    if len(q) > 0:
        break
    for y in wins[x]:
        loses[y] -= 1
        if loses[y] == 0:
            q.append(y)

print(x+1 if len(q) == 0 else -1)