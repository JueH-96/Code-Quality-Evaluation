def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    # convert to zero-indexed; a[i] represents the neighbor of i.
    a = list(map(lambda x: int(x) - 1, data[1:]))
    
    # dp[i]: number of vertices reachable from vertex i (including itself)
    dp = [-1] * n
    # visited marker: 0 = unvisited, 1 = currently in recursion stack, 2 = finished (dp computed)
    visited = [0] * n

    # We perform a DFS style cycle detection. In a graph where every vertex has outdegree 1,
    # each vertex eventually falls in some cycle.
    def dfs(u):
        stack = []  # track the current path
        cur = u
        while True:
            if visited[cur] == 0:
                visited[cur] = 1
                stack.append(cur)
                # Move along the unique edge.
                cur = a[cur]
                continue
            elif visited[cur] == 1:
                # A cycle is detected. Identify nodes in the cycle.
                # Everything from the first occurrence of cur in the stack to the end forms the cycle.
                cycle_start_index = stack.index(cur)
                cycle_nodes = stack[cycle_start_index:]
                cycle_len = len(cycle_nodes)
                # For every node in the cycle, its reachable set is exactly all nodes in the cycle.
                for node in cycle_nodes:
                    dp[node] = cycle_len
                    visited[node] = 2
                # For nodes that are in the chain before the cycle, their dp is one plus the dp of their child.
                for node in reversed(stack[:cycle_start_index]):
                    dp[node] = dp[a[node]] + 1
                    visited[node] = 2
                break
            else:
                # visited[cur] == 2, so dp[cur] is already computed.
                # Unwind the stack and assign dp[node] = dp[a[node]] + 1.
                while stack:
                    node = stack.pop()
                    dp[node] = dp[a[node]] + 1
                    visited[node] = 2
                break

    for i in range(n):
        if visited[i] == 0:
            dfs(i)
            
    # The answer is the sum over all vertices of the number of reachable vertices.
    answer = sum(dp)
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()