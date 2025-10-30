import sys
    input = sys.stdin.readline
    N = int(input())
    A = list(map(lambda x:int(x)-1, input().split()))
    e = [[] for _ in range(N)]
    for i in range(N):
        e[i].append(A[i])
    scc = Kosaraju(N, e, e).scc()
    
    ans = 0
    prev_scc = None
    for i in range(N):
        if prev_scc == scc[i]:
            continue
        ans += (N - scc[i] + 1) * scc[i] + scc[i] * (scc[i] - 1) // 2
        prev_scc = scc[i]

    print(ans)

class Kosaraju:
    def __init__(self, n, edges, reverse_edges, root=0):
        self.n, self.edges = n, edges
        self.post_order = self.dfs(self.edges, root)
        rev_map = {i:i for i in range(n)}
        for k,v in enumerate(self.post_order):
            rev_map[v] = k
        self.group = self.dfs(reverse_edges, root, map = rev_map)

    def traversal(self, edges, root, map = None):
        if map is None:
            map = {i:i for i in range(self.n)}

        dist = {}
        stack = [root]
        dist[root] = len(dist)
        travelled = [root]
        while stack:
            v = stack[-1]
            if v in edges:
                for w in edges[v]:
                    if w not in dist:
                        dist[w] = len(dist)
                        stack.append(w)
                        travelled.append(w)
                        break
                else:
                    stack.pop()
            else:
                stack.pop()
        return [map[v] for v in travelled]

    def dfs(self, edges, root, map = None):
        loop = False
        dist = {}
        stack = [root]
        dist[root] = len(dist)
        while stack:
            v = stack[-1]
            if v in edges:
                current = edges[v].pop(0)
                if current not in dist:
                    stack.append(current)
                    dist[current] = len(dist)
                elif map is None and dist[current] < dist[v]:
                    loop = True
                elif map is not None and map[current] < map[v]:
                    loop = True
                if len(edges[v]) == 0:
                    stack.pop()
            else:
                stack.pop()

        if map is None or not loop:
            return self.traversal(edges, root)
        group = [None] * self.n
        for k, v in dist.items():
            group[k] = v
        return group

    def scc(self):
        ans = [None]*self.n
        group = set()
        for k, v in enumerate(self.group):
            if ans[v] is None:
                group.add(v)
                ans[k] = v
            else:
                ans[k] = ans[v]

        taken = set()
        for g in group:
            if g not in taken:
                taken.add(g)
                idx = [i for i, x in enumerate(ans) if x == g]
                for i in idx:
                    group.add(i)
                    taken.add(i)
                    ans[i] = min(ans[i], g)
        inv_label = {}
        label = 0
        for v in sorted(ans):
            if v not in inv_label:
                inv_label[v] = label
                label += 1

        return [inv_label[v] for v in ans]