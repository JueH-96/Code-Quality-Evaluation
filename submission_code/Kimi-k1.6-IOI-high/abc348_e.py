import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    edges = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a = int(data[idx])
        b = int(data[idx + 1])
        edges[a].append(b)
        edges[b].append(a)
        idx += 2
    
    C = list(map(int, data[idx:idx+N]))
    C = [0] + C  # 1-based indexing
    
    # Build rooted tree with BFS
    root = 1
    parent = [-1] * (N + 1)
    children = [[] for _ in range(N + 1)]
    visited = [False] * (N + 1)
    q = deque([root])
    visited[root] = True
    parent[root] = -1
    
    while q:
        u = q.popleft()
        for v in edges[u]:
            if not visited[v] and v != parent[u]:
                visited[v] = True
                parent[v] = u
                children[u].append(v)
                q.append(v)
    
    # Post-order traversal to compute sum_C and sum_d
    sum_C = [0] * (N + 1)
    sum_d = [0] * (N + 1)
    stack = [(root, False)]
    
    while stack:
        node, visited_flag = stack.pop()
        if not visited_flag:
            stack.append((node, True))
            # Push children in reverse order to process them in order
            for child in reversed(children[node]):
                stack.append((child, False))
        else:
            sum_C[node] = C[node]
            sum_d[node] = 0
            for child in children[node]:
                sum_C[node] += sum_C[child]
                sum_d[node] += sum_d[child] + sum_C[child]
    
    total_C = sum_C[root]
    f_val = [0] * (N + 1)
    f_val[root] = sum_d[root]
    
    # BFS to compute f_val for all nodes
    q = deque([root])
    parent_bfs = [-1] * (N + 1)
    parent_bfs[root] = -1
    
    while q:
        u = q.popleft()
        for v in edges[u]:
            if v != parent_bfs[u]:  # Not the parent in BFS traversal
                if parent_bfs[v] == -1:
                    parent_bfs[v] = u
                    f_val[v] = f_val[u] + (total_C - 2 * sum_C[v])
                    q.append(v)
    
    print(min(f_val[1:N+1]))

if __name__ == "__main__":
    main()