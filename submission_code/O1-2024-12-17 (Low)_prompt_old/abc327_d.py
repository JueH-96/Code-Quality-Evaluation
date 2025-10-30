def solve():
    import sys
    sys.setrecursionlimit(10**7)
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+M]))
    B = list(map(int, data[2+M:2+2*M]))
    
    # Build adjacency
    # We want to check whether the graph defined by edges (A[i], B[i]) is bipartite.
    adj = [[] for _ in range(N+1)]
    for i in range(M):
        adj[A[i]].append(B[i])
        adj[B[i]].append(A[i])
    
    color = [-1]*(N+1)
    
    # We can do a DFS or BFS to check bipartiteness on each connected component.
    def dfs(start):
        stack = [start]
        color[start] = 0  # start with color 0
        while stack:
            node = stack.pop()
            for nxt in adj[node]:
                if color[nxt] == -1:
                    color[nxt] = 1 - color[node]
                    stack.append(nxt)
                elif color[nxt] == color[node]:
                    # conflict detected, not bipartite
                    return False
        return True
    
    for i in range(1, N+1):
        if color[i] == -1:
            if not dfs(i):
                print("No")
                return
                
    print("Yes")


def main():
    solve()

if __name__ == "__main__":
    main()