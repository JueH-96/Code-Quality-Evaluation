import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr]) - 1  # convert to 0-based
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    P = list(map(lambda x: int(x)-1, input[ptr:ptr+N]))  # convert to 0-based
    ptr += N
    Q = list(map(lambda x: int(x)-1, input[ptr:ptr+N]))  # convert to 0-based
    ptr += N
    
    in_graph = [False] * N
    adj = [[] for _ in range(N)]
    
    for i in range(N):
        if A[i] == 1:
            current = i
            path = []
            visited = set()
            while True:
                if current == X:
                    break
                if current in visited:
                    print(-1)
                    return
                visited.add(current)
                path.append(current)
                current = P[current]
            for node in path:
                in_graph[node] = True
            for u, v in zip(path, path[1:]):
                adj[u].append(v)
        
        if B[i] == 1:
            current = i
            path = []
            visited = set()
            while True:
                if current == X:
                    break
                if current in visited:
                    print(-1)
                    return
                visited.add(current)
                path.append(current)
                current = Q[current]
            for node in path:
                in_graph[node] = True
            for u, v in zip(path, path[1:]):
                adj[u].append(v)
    
    in_degree = [0] * N
    for u in range(N):
        if in_graph[u]:
            for v in adj[u]:
                in_degree[v] += 1
    
    queue = deque()
    for u in range(N):
        if in_graph[u] and in_degree[u] == 0:
            queue.append(u)
    
    processed = 0
    while queue:
        u = queue.popleft()
        processed += 1
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if processed == sum(in_graph):
        print(sum(in_graph))
    else:
        print(-1)

if __name__ == "__main__":
    main()