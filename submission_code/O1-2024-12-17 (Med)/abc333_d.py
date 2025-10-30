def main():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque
    
    input = sys.stdin.readline
    N = int(input())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, input().split())
        edges[u].append(v)
        edges[v].append(u)
    
    # If vertex 1 has degree 0 or 1, it's already a leaf (degree 0 won't happen given constraints, but we handle <=1 anyway).
    if len(edges[1]) <= 1:
        print(1)
        return
    
    # Otherwise, we remove "1" from consideration (mark it visited),
    # then find sizes of the components that arise from each neighbor of 1.
    visited = [False]*(N+1)
    visited[1] = True
    
    max_subtree_size = 0
    
    # BFS/DFS for each neighbor of 1, ignoring 1 (since it's visited),
    # to find how many vertices lie in that component.
    for neighbor in edges[1]:
        if not visited[neighbor]:
            queue = deque([neighbor])
            visited[neighbor] = True
            size_comp = 0
            while queue:
                node = queue.popleft()
                size_comp += 1
                for nxt in edges[node]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        queue.append(nxt)
            max_subtree_size = max(max_subtree_size, size_comp)
    
    # Minimum operations = N - size_of_largest_subtree_from_1
    # (this count includes removing 1 last).
    print(N - max_subtree_size)

# Do not forget to call main().
main()