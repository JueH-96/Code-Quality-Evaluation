def find_root(x, parent):
    if parent[x] != x:
        parent[x] = find_root(parent[x], parent)
    return parent[x]

def union(x, y, parent, rank):
    px = find_root(x, parent)
    py = find_root(y, parent)
    if px == py:
        return False
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1
    return True

def get_all_spanning_trees(n, edges):
    def dfs(curr_edges, used_edges, parent, rank):
        if len(used_edges) == n-1:
            trees.append(used_edges[:])
            return
            
        for i in range(curr_edges, len(edges)):
            u, v, w = edges[i]
            new_parent = parent[:]
            new_rank = rank[:]
            if union(u, v, new_parent, new_rank):
                used_edges.append(i)
                dfs(i+1, used_edges, new_parent, new_rank)
                used_edges.pop()

    trees = []
    parent = list(range(n+1))
    rank = [0] * (n+1)
    dfs(0, [], parent, rank)
    return trees

N, M, K = map(int, input().split())
edges = []
for _ in range(M):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

min_cost = float('inf')
trees = get_all_spanning_trees(N, edges)

for tree in trees:
    cost = 0
    for edge_idx in tree:
        cost = (cost + edges[edge_idx][2]) % K
    min_cost = min(min_cost, cost)

print(min_cost)