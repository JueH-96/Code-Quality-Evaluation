import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    from collections import defaultdict
    testimonies = defaultdict(list)
    for _ in range(M):
        A, B, C = map(int, sys.stdin.readline().split())
        testimonies[A].append((B, C))
    
    num_literals = 2 * N
    graph = [[] for _ in range(num_literals)]
    
    for A in testimonies:
        lst = testimonies[A]
        if not lst:
            continue
        B1, C1 = lst[0]
        if C1 == 0:
            desired1_lit = 2 * (B1 - 1)
        else:
            desired1_lit = 2 * (B1 - 1) + 1
        for (Bj, Cj) in lst[1:]:
            if Cj == 0:
                desiredj_lit = 2 * (Bj - 1)
            else:
                desiredj_lit = 2 * (Bj - 1) + 1
            a = desiredj_lit ^ 1
            b = desired1_lit
            graph[a].append(b)
            a = desired1_lit ^ 1
            b = desiredj_lit
            graph[a].append(b)
    
    def kosaraju(n, adj):
        visited = [False] * n
        order = []
        component = [-1] * n
        
        def dfs(u):
            stack = [(u, False)]
            while stack:
                node, processed = stack.pop()
                if processed:
                    order.append(node)
                    continue
                if visited[node]:
                    continue
                visited[node] = True
                stack.append((node, True))
                for v in adj[node]:
                    if not visited[v]:
                        stack.append((v, False))
        
        for u in range(n):
            if not visited[u]:
                dfs(u)
        
        visited = [False] * n
        comp = 0
        for u in reversed(order):
            if not visited[u]:
                stack = [u]
                visited[u] = True
                component[u] = comp
                while stack:
                    v = stack.pop()
                    for w in adj[v]:
                        if not visited[w]:
                            visited[w] = True
                            component[w] = comp
                            stack.append(w)
                comp += 1
        return component, comp
    
    component, comp_count = kosaraju(num_literals, graph)
    assignment = []
    for x in range(N):
        a = 2 * x
        b = 2 * x + 1
        if component[a] == component[b]:
            print(-1)
            return
        assignment.append(component[a] > component[b])
    
    c = [0] * N
    for A in range(1, N + 1):
        if A not in testimonies:
            continue
        lst = testimonies[A]
        B, C = lst[0]
        h_B = assignment[B - 1]
        if C == 0:
            desired = h_B
        else:
            desired = not h_B
        h_A = assignment[A - 1]
        c_val = (h_A) ^ desired
        c[A - 1] = c_val
    
    output = ''.join(map(str, c))
    print(output)

if __name__ == "__main__":
    main()