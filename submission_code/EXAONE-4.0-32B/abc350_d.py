import collections
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    m = int(data[1])
    graph = [[] for _ in range(n+1)]
    idx = 2
    for _ in range(m):
        u = int(data[idx])
        v = int(data[idx+1])
        idx += 2
        graph[u].append(v)
        graph[v].append(u)
    
    visited = [False] * (n+1)
    total_edges_after = 0
    
    for i in range(1, n+1):
        if not visited[i]:
            queue = collections.deque([i])
            visited[i] = True
            comp_size = 0
            while queue:
                node = queue.popleft()
                comp_size += 1
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            total_edges_after += comp_size * (comp_size - 1) // 2
    
    ans = total_edges_after - m
    print(ans)

if __name__ == "__main__":
    main()