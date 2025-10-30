def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    idx = 1
    edges = []
    adj = [[] for _ in range(N+1)]
    degree = [0] * (N+1)
    
    sum_edges = 0
    for _ in range(N-1):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        idx += 3
        adj[A].append((B, C))
        adj[B].append((A, C))
        degree[A] += 1
        degree[B] += 1
        sum_edges += C
    
    # Collect smallest edge costs for each leaf
    leaf_costs = []
    for node in range(1, N+1):
        if degree[node] == 1:
            # It's a leaf
            neighbor, cost = adj[node][0]
            leaf_costs.append(cost)
    
    sum_leaves = sum(leaf_costs)
    if len(leaf_costs) < 2:
        # Should not happen for N >= 2
        print(sum_edges + sum_leaves)
        return
    
    # Sort leaf costs in descending order
    leaf_costs.sort(reverse=True)
    # Sum of the two largest leaf costs
    sum_two_largest = leaf_costs[0] + leaf_costs[1]
    
    # Calculate minimal travel distance
    min_travel_distance = sum_edges + sum_leaves - sum_two_largest
    print(min_travel_distance)

if __name__ == "__main__":
    main()