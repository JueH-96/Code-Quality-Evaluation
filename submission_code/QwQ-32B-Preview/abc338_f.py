import sys
import math

def main():
    import sys
    import math

    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N)]
    for _ in range(M):
        U, V, W = map(int, sys.stdin.readline().split())
        U -= 1  # Convert to 0-based index
        V -= 1
        adj[U].append((V, W))

    # Function to check if the graph is strongly connected
    def is_strongly_connected(N, adj):
        # Check if all vertices are reachable from vertex 0
        visited = [False] * N
        stack = [0]
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                for v, _ in adj[u]:
                    if not visited[v]:
                        stack.append(v)
        if not all(visited):
            return False
        # Check if all vertices can reach vertex 0
        visited = [False] * N
        stack = [0]
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                for v, _ in adj[u]:
                    if not visited[v]:
                        stack.append(v)
        return all(visited)

    if not is_strongly_connected(N, adj):
        print("No")
        return

    # Initialize DP table
    dp = [[math.inf] * N for _ in range(1 << N)]
    for u in range(N):
        dp[1 << u][u] = 0

    # Iterate over all subsets
    for state in range(1 << N):
        for u in range(N):
            if not (state & (1 << u)):
                continue
            for v, w in adj[u]:
                if state & (1 << v):
                    dp[state][v] = min(dp[state][v], dp[state][u] + w)
                else:
                    new_state = state | (1 << v)
                    dp[new_state][v] = min(dp[new_state][v], dp[state][u] + w)

    # Find the minimum cost to visit all vertices
    result = min(dp[(1 << N) - 1])
    if result == math.inf:
        print("No")
    else:
        print(result)

if __name__ == "__main__":
    main()