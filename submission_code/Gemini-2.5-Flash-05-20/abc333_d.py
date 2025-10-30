import sys

# Set recursion limit for deep trees
sys.setrecursionlimit(3 * 10**5 + 100)

def solve():
    N = int(sys.stdin.readline())

    adj = [[] for _ in range(N + 1)]
    initial_degrees = [0] * (N + 1)

    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        initial_degrees[u] += 1
        initial_degrees[v] += 1

    # Base case: If vertex 1 is a leaf initially.
    # For N >= 2, if degree of 1 is 1, it's a leaf.
    if initial_degrees[1] == 1:
        print(1)
        return

    # dfs(u, p) calculates the minimum operations to delete all vertices in the subtree
    # rooted at u, treating p as its parent.
    # The return value represents the 'cost' or 'depth' of operations to clear this branch.
    def dfs(u, p):
        # Collect costs from all children subtrees
        child_costs = []
        for v in adj[u]:
            if v == p:
                continue # Skip the parent
            child_costs.append(dfs(v, u))

        # Sort costs in descending order. This is crucial for the greedy strategy.
        # The idea is that we peel branches. The 'i' in 'cost + i' accounts for
        # the fact that we have to wait for 'i' other branches (that are equally
        # or more 'expensive' to clear) to be processed before this one.
        child_costs.sort(reverse=True)

        max_ops_for_children = 0
        for i, cost in enumerate(child_costs):
            # 'cost + i' represents the time step at which this branch (cost) would be cleared
            # if we process other branches first (indicated by 'i').
            # We take the maximum to find the bottleneck.
            max_ops_for_children = max(max_ops_for_children, cost + i)
        
        # Add 1 for deleting the current node 'u' itself.
        return max_ops_for_children + 1

    # Main logic: Calculate costs for all branches connected to vertex 1.
    # Vertex 1 has degree D. To make it a leaf, D-1 of its branches must be cleared.
    # We choose the D-1 branches with the smallest 'dfs' costs to clear.
    # The 1 operation for deleting vertex 1 itself is added at the end.
    
    neighbor_branch_costs = []
    for neighbor in adj[1]:
        # Calculate the cost to clear the branch starting at 'neighbor', treating 1 as its parent.
        neighbor_branch_costs.append(dfs(neighbor, 1))

    # Sort these costs in ascending order to easily pick the smallest (D-1) branches.
    neighbor_branch_costs.sort()

    total_operations = 1 # Start with 1 for deleting vertex 1 itself

    # Sum the costs of the (initial_degrees[1] - 1) branches that must be fully cleared.
    # These are the ones with the smallest costs, as per the sorted list.
    # The one branch with the highest cost effectively becomes the 'main path'
    # that eventually leads to 1 being a leaf, and its nodes are implicitly handled
    # as part of the total operations but not as a separate sum.
    
    # We loop through initial_degrees[1] - 1 elements, which are the smallest costs.
    # initial_degrees[1] is guaranteed to be >= 2 due to the base case check.
    for i in range(initial_degrees[1] - 1):
        total_operations += neighbor_branch_costs[i]
    
    print(total_operations)

solve()