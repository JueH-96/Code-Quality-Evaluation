from collections import deque
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    
    # Build adjacency list (1-based)
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        a = int(data[idx])
        b = int(data[idx+1])
        adj[a].append(b)
        adj[b].append(a)
        idx += 2
    
    C = list(map(int, data[idx:idx + n]))
    idx += n
    
    # Compute sum_sub using iterative post-order traversal
    sum_sub = [0] * (n + 1)
    root = 1
    stack = [(root, None, False)]
    while stack:
        node, parent, visited = stack.pop()
        if not visited:
            stack.append((node, parent, True))
            # Push children (neighbors except parent)
            for neighbor in adj[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, False))
        else:
            sum_sub[node] = C[node - 1]  # C is 0-based
            for neighbor in adj[node]:
                if neighbor != parent:
                    sum_sub[node] += sum_sub[neighbor]
    
    # Compute f_root using BFS
    distance = {root: 0}
    f_root = 0
    queue = deque([root])
    while queue:
        node = queue.popleft()
        f_root += C[node - 1] * distance[node]
        for neighbor in adj[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    
    total_C = sum(C)
    min_f = f_root
    f = {root: f_root}
    queue = deque([(root, None)])
    
    while queue:
        u, p = queue.popleft()
        for v in adj[u]:
            if v == p:
                continue
            f_v = f[u] + (total_C - 2 * sum_sub[v])
            if f_v < min_f:
                min_f = f_v
            f[v] = f_v
            queue.append((v, u))
    
    print(min_f)

if __name__ == '__main__':
    main()