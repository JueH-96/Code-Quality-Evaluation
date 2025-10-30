from collections import defaultdict

def dfs(v, parent):
    global bfs_order, memo
    for u in graph[v]:
        if u != parent:
            bfs_order.append(u)
            memo[u] = len(bfs_order) - 1
            dfs(u, v)

def reverse_build(v, parent, component):
    global am, sigma_an, amortized
    for u in graph[v]:
        if u != parent:
            if child[u]:
                reverse_build(u, v, component)
            else:
                a = am[v] + 1
                am[u] = a
                sigma_an[component] += a * (an[v] + 1)
                amortized[v] += a * (an[v] + 1)

N = int(input())
graph = defaultdict(set)

for _ in range(N - 1):
    u, v = map(int, input().split())
    graph[u].add(v)
    graph[v].add(u)

degs = [len(graph[v]) for v in graph]

dists = defaultdict(list)
bfs_order = [1]
memo = {1: 0}
dfs(1, None)

splitter = [None] * (len(bfs_order) + 1)
for i in range(len(bfs_order) - 1, -1, -1):
    v = bfs_order[i]
    p = memo[v]
    make = True
    splits = 0
    for u in graph[v]:
        if memo[u] < p:
            splits += 1
            if splits > 1:
                make = False
                break
    if make:
        splitter[p] = i
for i, s in enumerate(splitter[1:], 1):
    if s is None:
        splitter[i] = splitter[i - 1]
dist = {}
paths = []
started = False
component = 0
am = [None] * (N + 1)
sigma_an = [0] * (N + 1)
amortized = [None] * (N + 1)
an = [None] * (N + 1)
stack = []
child = [False] * (N + 1)
dist[1] = 0
paths.append([1])
an[1] = degs[1] - 1
started = True
for i in range(len(bfs_order)):
    v = bfs_order[i]
    if started:
        add = []
        for u in graph[v]:
            if memo[u] < memo[v]:
                stack.append(u)
            elif memo[u] > memo[v] and not child[u]:
                add.append(u)
        if add:
            u = add[0]
            component += 1
            an[u] = 1
            am[u] = 0
            sigma_an[component] = 0
            amortized[u] = 0
            reverse_build(u, v, component)
        for u in add:
            child[u] = True
            paths[-1].append(u)
            if degs[u] > 1:
                dist[u] = degs[u] - 1
                dists[degs[u] - 1].append(u)
            else:
                reverse_build(u, v, component)
                dist[u] = am[u] + 1
                dists[am[u] + 1].append(u)
        for u in stack:
            if u not in add:
                paths.append([u])
                dist[u] = degs[u] - 1 + degs[v] - 1
                dists[degs[u] - 1 + degs[v] - 1].append(u)
                stack.pop()
                break
    if splitter[i+1] == i:
        started = False
        for u in dists[degs[v] - 1]:
            if amortized[v] + sigma_an[component] < amortized[u]:
                amortized[u] = amortized[v] + sigma_an[component]
        dists[degs[v] - 1] = []
        for u in dists[dist[v]]:
            if amortized[u] > amortized[v] + sigma_an[component] + 1:
                amortized[u] = amortized[v] + sigma_an[component] + 1
        dists[dist[v]] = []
        component = 0

levels = defaultdict(int)
for i in range(1, N+1):
    if amortized[i] is not None:
        levels[amortized[i]]. += 1

print(' '.join(str(i) for _ in range(levels) for i in range(1, levels+1)))