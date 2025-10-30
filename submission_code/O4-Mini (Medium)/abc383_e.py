import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin
    line = data.readline().split()
    N = int(line[0]); M = int(line[1]); K = int(line[2])
    edges = []
    for _ in range(M):
        u,v,w = data.readline().split()
        edges.append((int(w), int(u), int(v)))
    # Read A and B lists
    A_list = list(map(int, data.readline().split()))
    B_list = list(map(int, data.readline().split()))
    # Sort edges by weight
    edges.sort()
    # DSU for building Kruskal reconstruction tree
    parent = list(range(N+1))
    size = [1] * (N+1)
    # node_rep[x] = the current tree-node representing the component whose DSU root is x
    node_rep = list(range(N+1))
    def find(x):
        # path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    # Allocate arrays for reconstruction tree
    # We know there will be exactly N-1 merges, so final nodes = N + (N-1) = 2N-1
    # allocate a bit more
    max_nodes = 2 * N + 5
    weight = [0] * max_nodes
    child1 = [0] * max_nodes
    child2 = [0] * max_nodes
    # idx is the next new node id
    idx = N
    # Kruskal: build the merge tree
    for w,u,v in edges:
        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue
        # representatives
        nu = node_rep[ru]
        nv = node_rep[rv]
        idx += 1
        weight[idx] = w
        child1[idx] = nu
        child2[idx] = nv
        # union by size
        if size[ru] < size[rv]:
            parent[ru] = rv
            size[rv] += size[ru]
            new_root = rv
        else:
            parent[rv] = ru
            size[ru] += size[rv]
            new_root = ru
        # map the new component to the new tree-node
        node_rep[new_root] = idx
    # rem_A and rem_B arrays to count leftover A/B in each subtree
    rem_A = [0] * (idx+1)
    rem_B = [0] * (idx+1)
    # initialize leaves
    for a in A_list:
        rem_A[a] += 1
    for b in B_list:
        rem_B[b] += 1
    total = 0
    # bottom-up: nodes N+1 .. idx; children always < u
    for u in range(N+1, idx+1):
        c1 = child1[u]
        c2 = child2[u]
        # total A and B in this subtree before matching here
        a_cnt = rem_A[c1] + rem_A[c2]
        b_cnt = rem_B[c1] + rem_B[c2]
        # match as many as possible here
        m = a_cnt if a_cnt < b_cnt else b_cnt
        if m:
            total += m * weight[u]
        # leftovers bubble up
        rem_A[u] = a_cnt - m
        rem_B[u] = b_cnt - m
    # print the answer
    print(total)

if __name__ == "__main__":
    main()