import sys
import threading

def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    A = [0] * M
    B = [0] * M
    for i in range(M):
        A[i] = int(next(it))
    for i in range(M):
        B[i] = int(next(it))
    
    # If any self-loop, impossible
    for i in range(M):
        if A[i] == B[i]:
            print("No")
            return
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for u, v in zip(A, B):
        adj[u].append(v)
        adj[v].append(u)
    
    # -1 = uncolored, 0 or 1 = colors
    color = [-1] * (N+1)
    
    # BFS to 2-color each component
    for start in range(1, N+1):
        if color[start] != -1 or not adj[start]:
            # already colored or isolated node (no constraints)
            continue
        # start BFS
        dq = deque([start])
        color[start] = 0
        while dq:
            u = dq.popleft()
            cu = color[u]
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = cu ^ 1
                    dq.append(v)
                elif color[v] == cu:
                    # conflict
                    print("No")
                    return
    
    # If no conflicts found, it's possible
    print("Yes")

if __name__ == "__main__":
    main()