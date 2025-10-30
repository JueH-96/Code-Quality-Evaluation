import sys
from sys import stdin

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    ops = []
    for i in range(Q):
        L, R, C = map(int, sys.stdin.readline().split())
        ops.append((C, L, R, i + 1))  # i+1 is the auxiliary node index
    # Sort ops by C in increasing order
    ops.sort()
    # Initialize DSU for N + Q nodes
    max_node = N + Q
    parent = list(range(max_node + 2))  # nodes are 1..N+Q
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        # Merge smaller rank into larger rank
        parent[v_root] = u_root
        return True
    # Initialize jump pointers for original nodes
    jump = list(range(1, N + 2))  # jump[j] for j=1..N
    total_cost = 0
    # Process each operation in sorted order
    for C, L, R, i in ops:
        aux_node = N + i
        current = L
        while current <= R:
            if current > N:
                break
            root = find(current)
            if root > N:  # part of an auxiliary component
                current = jump[root]
                continue
            if find(root) == find(aux_node):
                current = jump[root]
                continue
            # Merge the original component with the auxiliary node
            if union(root, aux_node):
                total_cost += C
            # Update the jump pointer for the merged component
            new_root = find(aux_node)
            # Set the jump pointer of the new_root to the maximum of the original jump
            jump[new_root] = max(jump[new_root], jump[root])
            current = jump[root]
    # Check if all nodes are connected
    root = find(1)
    for i in range(2, N + Q + 1):
        if find(i) != root:
            print(-1)
            return
    print(total_cost)

main()