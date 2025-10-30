# YOUR CODE HERE
import sys

def solve():
    n, q = map(int, sys.stdin.readline().split())
    queries = []
    for _ in range(q):
        l, r, c = map(int, sys.stdin.readline().split())
        queries.append((l, r, c))

    parent = list(range(n + q + 1))
    rank = [0] * (n + q + 1)
    mst_cost = 0

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j, cost):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            nonlocal mst_cost
            mst_cost += cost
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1
            return True
        return False

    for i in range(q):
        l, r, c = queries[i]
        for j in range(l, r + 1):
            union(n + i + 1, j, c)

    num_components = 0
    for i in range(1, n + 1):
        if parent[i] == i:
            num_components += 1
    
    if num_components > 1:
        print(-1)
    else:
        print(mst_cost)


solve()