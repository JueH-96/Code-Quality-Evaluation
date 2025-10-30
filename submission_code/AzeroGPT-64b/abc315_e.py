from collections import deque

n = int(input())
g = [[] for _ in range(n)]
deg = [0] * n
rev = [[] for _ in range(n)]
ans = [0] * n

for v in range(n):
    c, *p = list(map(int, input().split()))
    for pi in p:
        pi -= 1
        g[pi].append(v)
        deg[v] += 1
    rev[v] = p

q = deque([])
_TODO = 0
_DONE = 1
for v in range(n):
    if not deg[v]: q.append((v, _DONE))
pq = [[], []]
while q:
    v, state = q.popleft()
    if state == _TODO:
        for w in range(n):
            if v in rev[w]:
                rev[w].remove(v)
                if not rev[w]:
                    ans[w] = v + 1
                    if w > 0: pq[0].append(w)
                    else: pq[1].append(w)

        for w in g[v]:
            deg[w] -= 1
            if not deg[w]: q.append((w, _DONE))
    else:
        q.append((v, _TODO))
        for w in pq[0]:
            q.appendleft((w, _DONE))
        pq = pq[1:]
        for w in pq[0]:
            q.append((w, _TODO))

print(*ans[1:])