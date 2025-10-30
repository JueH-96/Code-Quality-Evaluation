from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    reversed_graph = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        C_i = int(input[ptr])
        ptr += 1
        P_i = list(map(int, input[ptr:ptr + C_i]))
        ptr += C_i
        reversed_graph[i] = P_i
    
    required_nodes = set()
    queue = deque([1])
    required_nodes.add(1)
    while queue:
        u = queue.popleft()
        for v in reversed_graph[u]:
            if v not in required_nodes:
                required_nodes.add(v)
                queue.append(v)
    
    required = list(required_nodes - {1})
    if not required:
        print()
        return
    
    adj = [[] for _ in range(N + 1)]
    in_degree = [0] * (N + 1)
    for i in required:
        for p in reversed_graph[i]:
            adj[p].append(i)
            in_degree[i] += 1
    
    queue = deque()
    for i in required:
        if in_degree[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    print(' '.join(map(str, order)))

if __name__ == "__main__":
    main()