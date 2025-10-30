# YOUR CODE HERE
import sys

import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 20)
    N, M = map(int, sys.stdin.readline().split())
    op_list = []
    total_nodes = 0
    for _ in range(M):
        K_C = sys.stdin.readline().split()
        while len(K_C) < 2:
            K_C += sys.stdin.readline().split()
        K_i, C_i = int(K_C[0]), int(K_C[1])
        nodes_line = []
        while len(nodes_line) < K_i:
            nodes_line += sys.stdin.readline().split()
        S_i = list(map(int, nodes_line))
        total_nodes += K_i
        op_list.append( (C_i, S_i) )

    op_list.sort()

    parent = [i for i in range(N+1)]  # 1-indexed
    rank = [0]*(N+1)
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        # Union by rank
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[v_root] = u_root
            if rank[u_root] == rank[v_root]:
                rank[u_root] +=1
        return True

    num_components = N
    total_cost = 0

    for C_i, S_i in op_list:
        roots = set()
        for u in S_i:
            roots.add(find(u))
        roots = list(roots)
        if len(roots) > 1:
            first_root = roots[0]
            for r in roots[1:]:
                if union(first_root, r):
                    total_cost += C_i
                    num_components -= 1

    if num_components != 1:
        print(-1)
    else:
        print(total_cost)

threading.Thread(target=main).start()