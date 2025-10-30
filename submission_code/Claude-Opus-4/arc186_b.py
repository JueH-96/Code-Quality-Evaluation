def solve():
    MOD = 998244353
    
    n = int(input())
    a = list(map(int, input().split()))
    
    # Build a graph of constraints
    # greater[i] = list of positions that must have greater value than position i
    # less[i] = list of positions that must have less value than position i
    greater = [[] for _ in range(n)]
    less = [[] for _ in range(n)]
    
    for i in range(n):
        if a[i] > 0:
            # P_{A_i} < P_i
            less[i].append(a[i] - 1)
            greater[a[i] - 1].append(i)
        
        # For all j with A_i < j < i: P_j > P_i
        for j in range(a[i], i):
            greater[j].append(i)
            less[i].append(j)
    
    # Topological sort to find a valid ordering
    in_degree = [0] * n
    for i in range(n):
        for j in greater[i]:
            in_degree[j] += 1
    
    # Process in topological order
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in greater[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Count the number of valid permutations
    # We need to assign values 1 to n to positions
    # such that the constraints are satisfied
    
    # For each position in topological order, we can choose from available values
    # that respect the constraints
    
    # Actually, let's think differently
    # We process positions in order and for each position,
    # we determine how many valid values can be assigned
    
    # Build constraint graph more carefully
    adj = [set() for _ in range(n)]  # adj[i] contains positions that must have value > position i
    
    for i in range(n):
        for j in range(a[i], i):
            adj[i].add(j)
        if a[i] > 0:
            adj[a[i] - 1].add(i)
    
    # Convert to reverse adjacency for easier processing
    rev_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            rev_adj[j].add(i)
    
    # Dynamic programming approach
    # Process positions and count valid assignments
    
    # Actually, let's use a different approach
    # The key insight is that we need to count the number of linear extensions
    # of the partial order defined by the constraints
    
    # Use inclusion-exclusion or a different counting method
    
    # For now, let's implement a solution that works for the given constraints
    
    # The number of valid permutations can be computed by considering
    # the structure of the constraints
    
    result = 1
    used = [False] * n
    
    # Process positions in a specific order
    # and multiply by the number of choices at each step
    
    # This is a complex counting problem
    # Let me implement a solution based on the structure
    
    # For each connected component in the constraint graph,
    # we can count the number of valid orderings
    
    # Simplified approach for the specific problem structure
    from math import factorial
    
    # Group positions by their A values
    groups = {}
    for i in range(n):
        if a[i] not in groups:
            groups[a[i]] = []
        groups[a[i]].append(i)
    
    # Count valid permutations
    result = 1
    
    # Process based on the structure of A values
    # This is problem-specific logic based on the constraints
    
    # For the given problem, we need to carefully analyze the structure
    # and count the number of valid assignments
    
    # Let's implement a working solution
    dp = {}
    
    def count_ways(mask, last_used):
        if mask == (1 << n) - 1:
            return 1
        
        if (mask, last_used) in dp:
            return dp[(mask, last_used)]
        
        ways = 0
        for i in range(n):
            if mask & (1 << i):
                continue
            
            # Check if we can place value (last_used + 1) at position i
            valid = True
            
            # Check constraints
            for j in range(a[i], i):
                if not (mask & (1 << j)):
                    valid = False
                    break
            
            if a[i] > 0 and not (mask & (1 << (a[i] - 1))):
                valid = False
            
            if valid:
                ways = (ways + count_ways(mask | (1 << i), last_used + 1)) % MOD
        
        dp[(mask, last_used)] = ways
        return ways
    
    # This approach might be too slow for large n
    # Let's use a more efficient method
    
    # Based on the problem structure, implement an efficient solution
    
    # Count the number of valid topological orderings
    from collections import defaultdict
    
    # Build adjacency list for the DAG
    graph = defaultdict(list)
    indeg = [0] * n
    
    for i in range(n):
        for j in range(a[i], i):
            graph[j].append(i)
            indeg[i] += 1
        if a[i] > 0:
            graph[i].append(a[i] - 1)
            indeg[a[i] - 1] += 1
    
    # Count topological orderings
    def count_topo(remaining, indegree):
        if not remaining:
            return 1
        
        count = 0
        candidates = [v for v in remaining if indegree[v] == 0]
        
        for v in candidates:
            new_remaining = remaining - {v}
            new_indegree = indegree.copy()
            
            for u in graph[v]:
                if u in new_remaining:
                    new_indegree[u] -= 1
            
            count = (count + count_topo(new_remaining, new_indegree)) % MOD
        
        return count
    
    # This is still potentially slow
    # Let's implement a more efficient solution based on the problem structure
    
    # The key insight is that the answer depends on the structure of the A array
    # We need to count the number of ways to assign values respecting the constraints
    
    # For the given problem, let's implement a working solution
    print(count_topo(set(range(n)), indeg))

solve()