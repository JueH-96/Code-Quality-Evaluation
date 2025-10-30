from collections import deque
import sys

def main():
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = input_data[2:]
    
    # Adjacency list for the original graph and the reversed graph
    graph = [[] for _ in range(N+1)]
    graph_rev = [[] for _ in range(N+1)]
    
    idx = 0
    for _ in range(M):
        a = int(edges[idx]); b = int(edges[idx+1])
        idx += 2
        graph[a].append(b)
        graph_rev[b].append(a)
    
    # BFS from node 1 in the original graph
    dist_1 = [-1]*(N+1)
    dist_1[1] = 0
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if dist_1[v] == -1:
                dist_1[v] = dist_1[u] + 1
                queue.append(v)
    
    # BFS from node 1 in the reversed graph
    dist_rev = [-1]*(N+1)
    dist_rev[1] = 0
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for v in graph_rev[u]:
            if dist_rev[v] == -1:
                dist_rev[v] = dist_rev[u] + 1
                queue.append(v)
    
    # Find the minimum cycle containing vertex 1
    # A valid cycle can be formed by a path 1 -> ... -> u -> v -> ... -> 1 
    # if dist_1[u] != -1 and dist_rev[v] != -1 for an edge (u->v).
    # Cycle length = dist_1[u] + 1 + dist_rev[v]
    INF = 10**15
    ans = INF
    for u in range(1, N+1):
        if dist_1[u] == -1:
            continue
        for v in graph[u]:
            # If we can return from v to 1, we have a valid cycle
            if dist_rev[v] != -1:
                cycle_len = dist_1[u] + 1 + dist_rev[v]
                if cycle_len < ans:
                    ans = cycle_len
    
    print(ans if ans < INF else -1)

# Do not forget to call main()
if __name__ == "__main__":
    main()