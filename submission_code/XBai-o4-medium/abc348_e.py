import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a = int(input[ptr])
        b = int(input[ptr + 1])
        adj[a].append(b)
        adj[b].append(a)
        ptr += 2
    
    C = list(map(int, input[ptr:ptr + N]))
    ptr += N
    
    total_C = sum(C)
    
    sum_C_subtree = [0] * (N + 1)
    cost = [0] * (N + 1)
    
    # Post-order traversal to compute sum_C_subtree and cost
    stack = [(1, -1, False)]
    while stack:
        u, parent, visited = stack.pop()
        if not visited:
            stack.append((u, parent, True))
            # Push children in reverse order to process in original order (optional)
            for v in reversed(adj[u]):
                if v != parent:
                    stack.append((v, u, False))
        else:
            sum_c = C[u - 1]
            cost_u = 0
            for v in adj[u]:
                if v != parent:
                    sum_c += sum_C_subtree[v]
                    cost_u += cost[v] + sum_C_subtree[v]
            sum_C_subtree[u] = sum_c
            cost[u] = cost_u
    
    # BFS to compute f for all nodes
    f = [0] * (N + 1)
    f[1] = cost[1]
    q = deque()
    q.append((1, -1))
    
    while q:
        u, parent_u = q.popleft()
        for v in adj[u]:
            if v != parent_u:
                f[v] = f[u] + (total_C - 2 * sum_C_subtree[v])
                q.append((v, u))
    
    print(min(f[1:N+1]))

if __name__ == '__main__':
    main()