def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = list(map(int, data[1:N+1]))
    Y = list(map(int, data[N+1:2*N+1]))
    
    # Build dominance relations (DAG)
    from collections import defaultdict
    adj_list = defaultdict(list)
    for i in range(N):
        for j in range(N):
            if X[i] > X[j] and Y[i] > Y[j]:
                adj_list[i].append(j)
    
    # Compute topological order
    from collections import deque
    in_degree = [0] * N
    for i in adj_list:
        for j in adj_list[i]:
            in_degree[j] += 1
    queue = deque()
    for i in range(N):
        if in_degree[i] == 0:
            queue.append(i)
    top_order = []
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for neighbor in adj_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(top_order) != N:
        print("Error: DAG has cycles")
        return
    
    # DP to count order filters
    MOD = 998244353
    dp = [1] * N  # base case: each node can be in a filter by itself
    for node in reversed(top_order):
        with_node = 1  # include the node
        for neighbor in adj_list[node]:
            with_node = (with_node * dp[neighbor]) % MOD
        dp[node] = (with_node + 1) % MOD  # option to exclude the node
    
    # The total number of order filters is the product of choices for each connected component
    # In this DAG, it's the product of dp[node] for nodes with no incoming edges
    total = 1
    for node in range(N):
        if in_degree[node] == 0:
            total = (total * dp[node]) % MOD
    
    print(total)

if __name__ == "__main__":
    main()