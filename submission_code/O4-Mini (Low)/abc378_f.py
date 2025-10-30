import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    for _ in range(N-1):
        u,v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # We build the subgraph H of nodes whose original degree is 2 or 3.
    # In H we include an edge (u,v) iff it's an edge in the tree and both u,v have deg 2 or 3.
    used = [False]*(N+1)
    ans = 0

    for node in range(1, N+1):
        if not used[node] and (deg[node] == 2 or deg[node] == 3):
            # Start a DFS over this H-component
            stack = [node]
            used[node] = True
            comp_nodes = []
            while stack:
                u = stack.pop()
                comp_nodes.append(u)
                for w in adj[u]:
                    if not used[w] and (deg[w] == 2 or deg[w] == 3):
                        used[w] = True
                        stack.append(w)

            # Count how many deg-2 nodes in this component
            c2 = 0
            # Count how many edges in H connect two deg-2 nodes
            e22 = 0
            for u in comp_nodes:
                if deg[u] == 2:
                    c2 += 1
                    # look at its neighbors in H, count those also deg2
                    for w in adj[u]:
                        if deg[w] == 2:
                            e22 += 1
            # Each edge was counted twice in e22
            e22 //= 2

            # Number of ways to pick 2 distinct deg-2 nodes in this component
            # minus those that are directly adjacent in H (distance 1)
            # so that their path length >= 2
            # C(c2,2) - e22
            if c2 >= 2:
                ans += c2*(c2-1)//2 - e22

    print(ans)

def main_wrapper():
    main()

if __name__ == "__main__":
    main_wrapper()