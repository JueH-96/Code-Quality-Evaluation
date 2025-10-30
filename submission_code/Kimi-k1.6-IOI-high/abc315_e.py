from collections import deque
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N + 1)]
    reversed_adj = [[] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        C_i = int(input[ptr])
        ptr += 1
        P_list = list(map(int, input[ptr:ptr + C_i]))
        ptr += C_i
        adj[i] = P_list
        for v in P_list:
            reversed_adj[v].append(i)
    
    visited = [False] * (N + 1)
    q = deque()
    for v in adj[1]:
        if not visited[v]:
            visited[v] = True
            q.append(v)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    
    nodes_in_S = [i for i in range(1, N + 1) if visited[i]]
    sorted_nodes = sorted(nodes_in_S, reverse=True)
    
    in_degree = [0] * (N + 1)
    for i in range(1, N + 1):
        if visited[i]:
            in_degree[i] = len(adj[i])
    
    result = []
    q = deque()
    
    for u in sorted_nodes:
        if in_degree[u] == 0:
            q.append(u)
    
    while q:
        u = q.popleft()
        result.append(u)
        for v in reversed_adj[u]:
            if visited[v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()