import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    # Check if all boxes except X have zero balls
    has_non_zero = False
    for i in range(1, N+1):
        if i != X:
            if A[i-1] + B[i-1] > 0:
                has_non_zero = True
                break
    if not has_non_zero:
        print(0)
        return
    
    # Build closure set S
    visited = [False] * (N + 1)  # 1-based
    q = deque()
    for i in range(1, N+1):
        if i != X and (A[i-1] + B[i-1] > 0):
            if not visited[i]:
                visited[i] = True
                q.append(i)
    
    s_size = 0
    while q:
        current = q.popleft()
        s_size += 1
        p_val = P[current-1]
        if p_val != X and not visited[p_val]:
            visited[p_val] = True
            q.append(p_val)
        q_val = Q[current-1]
        if q_val != X and not visited[q_val]:
            visited[q_val] = True
            q.append(q_val)
    
    # Build the graph
    adj = [[] for _ in range(N+1)]
    in_degree = [0] * (N+1)
    for i in range(1, N+1):
        if visited[i]:
            a_val = A[i-1]
            if a_val > 0:
                p_val = P[i-1]
                if p_val != X:
                    adj[i].append(p_val)
                    in_degree[p_val] += 1
            b_val = B[i-1]
            if b_val > 0:
                q_val = Q[i-1]
                if q_val != X:
                    adj[i].append(q_val)
                    in_degree[q_val] += 1
    
    # Kahn's algorithm
    q_k = deque()
    for i in range(1, N+1):
        if visited[i] and in_degree[i] == 0:
            q_k.append(i)
    
    count = 0
    while q_k:
        u = q_k.popleft()
        count += 1
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q_k.append(v)
    
    if count == s_size:
        print(s_size)
    else:
        print(-1)

if __name__ == "__main__":
    main()