from collections import Counter, defaultdict

INF = 2**31

N, M = list(map(int, input().split()))
edges = [defaultdict(int) for _ in range(N)]
for i in range(M):
  u, v = list(map(int, input().split()))
  edges[u-1][v-1] = edges[v-1][u-1] = 1
W = list(map(int, input().split()))
items = list(map(int, input().split()))
headers = list(range(N))

vis = [0 for _ in range(N)]
def dfs(x):
  global headers, vis
  component = []
  stack = [x]
  gedges = defaultdict(list)
  has_edge = defaultdict(int)
  comp = []
  while stack:
    i = stack.pop()
    if vis[i]:
      continue
    vis[i] = 1
    component.append(i)
    comp.append(W[i])
    if W[i] == INF:
      headers[i] = x
      continue  
    headers[i] = x
    for j, _ in edges[i].items():
      if headers[j] == INF:
        headers[j] = i
        has_edge[j] = 1
      else:
        if headers[j] <= x:
          continue
      gedges[j].append(i)
      stack.append(j)
  totalLoad = sum(items[i] for i in component)
  comp.sort()
  while comp:
    cur = comp.pop()
    if any(has_edge[i] * W[i] < cur for i in component):
      continue
    headers = [x if h == x else h for h in headers]
    for i, neis in gedges.items():
      vis[i] = 0
      gedges[i] = [j for j in neis if headers[j] != x]
    stack = [i for i, isEdge in has_edge.items() if isEdge]
    dfs(x)
    totalLoad += sum(items[i] for i in component if headers[i] == x)
    break

dfs(INF)
ans = 0
for i in range(1, N):
  if headers[i] == i:
    continue
  ans += items[i]

print(ans)