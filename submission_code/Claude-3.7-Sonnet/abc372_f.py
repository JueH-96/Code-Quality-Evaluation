# YOUR CODE HERE
def count_ways(n, m, k, xs, ys):
    MOD = 998244353
    
    # If there are no Type 2 edges, there's only one possible sequence
    if m == 0:
        return 1
    
    # Initialize adjacency list
    graph = [[] for _ in range(n+1)]
    
    # Add Type 1 edges
    for i in range(1, n):
        graph[i].append(i+1)
    graph[n].append(1)  # From vertex N to vertex 1
    
    # Add Type 2 edges
    for x, y in zip(xs, ys):
        graph[x].append(y)
    
    # Count total ways using dynamic programming
    # dp_curr[j] represents the number of ways to reach vertex j after current steps
    # dp_prev[j] represents the number of ways to reach vertex j after previous steps
    dp_prev = [0] * (n+1)
    dp_prev[1] = 1  # Start at vertex 1
    
    for i in range(1, k+1):
        dp_curr = [0] * (n+1)
        for prev in range(1, n+1):
            if dp_prev[prev] == 0:
                continue  # Skip if there's no way to reach this vertex
            for next_vertex in graph[prev]:
                dp_curr[next_vertex] = (dp_curr[next_vertex] + dp_prev[prev]) % MOD
        dp_prev = dp_curr
    
    # Sum up total ways
    return sum(dp_prev[1:]) % MOD

def main():
    N, M, K = map(int, input().split())
    xs = []
    ys = []
    for _ in range(M):
        x, y = map(int, input().split())
        xs.append(x)
        ys.append(y)

    result = count_ways(N, M, K, xs, ys)
    print(result)

if __name__ == "__main__":
    main()