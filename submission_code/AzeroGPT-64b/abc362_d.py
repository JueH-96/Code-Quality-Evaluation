class Node:
    def __init__(self, h, index, weight, previous) -> None:
        self.h = h
        self.index = index
        self.weight = weight
        self.previous = previous

    def __lt__(self, other) -> bool:
        return self.weight < other.weight

    def __repr__(self) -> str:
        return f"x = {self.index}, h = {self.h}, weight = {self.weight}, previous = {self.previous.index}"
    
class MST:
    def __init__(self, edges, n) -> None:
        self.p = list(range(n))
        self.m = [0] * n
        self.edges = sorted(edges, key=lambda tup: tup[2])
        self.cnt = 0

    def par(self, x):
        if self.p[x] == x:
            return x
        self.p[x] = self.par(self.p[x])
        return self.p[x]

    def unite(self, i, j):
        pi, pj = self.par(i), self.par(j)
        if pi == pj:
            return
        if self.m[pi] > self.m[pj]:
            self.p[pj] = pi
        else:
            self.p[pi] = pj
            if self.m[pi] == self.m[pj]:
                self.m[pj] += 1
        self.cnt += 1

    def solve(self, edges):
        for i, (u, v, w) in enumerate(self.edges):
            if self.cnt == len(self.p) - 1:
                break
            self.unite(u, v)
            edges.append((u, v, w))
        return edges


def main():
    # read
    n, m = map(int, input().split())
    _a = list(map(int, input().split()))
    edges = [tuple([int(i) for i in input().split()]) for _ in range(m)]

    # construct MST
    tree = MST(edges, n).solve(edges)

    # prepare
    graph = [[] for _ in range(n)]
    start = 0
    for x, y, w in tree:
        graph[x - 1].append((y - 1, w))
        graph[y - 1].append((x - 1, w))
    
    # find path
    tree = [Node(1, i, _a[i], None) for i in range(n)]
    stack = [tree[0]]
    while stack:
        cur = stack.pop()
        
        for i, weight in graph[cur.index]:
            neighbor = tree[i]

            if neighbor.h == 0 and neighbor.weight > cur.weight + weight + _a[i]:
                neighbor.h = cur.h + 1
                neighbor.weight = cur.weight + weight + _a[i]
                neighbor.previous = cur
                stack.append(neighbor)
    
    # answer
    answer = [str(x.weight) for x in tree[1:]]
    print(" ".join(answer))

main()