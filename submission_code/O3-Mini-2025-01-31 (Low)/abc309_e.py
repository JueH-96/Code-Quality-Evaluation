def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Parse input
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Build tree from parent information
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        p = int(next(it))
        children[p].append(i)
    
    # For each person, store the events (insurances) that begin at that person.
    # Each event is characterized by a y value: it covers the node itself plus descendants 
    # for the next y generations. The event starting at node x gives a 'coverage limit' of depth[x] + y.
    events_by_node = [[] for _ in range(N+1)]
    for _ in range(M):
        x = int(next(it))
        y = int(next(it))
        events_by_node[x].append(y)
    
    # We can decide if a person u is covered by checking if there is an ancestor x 
    # (possibly u itself) that bought an insurance with parameter y such that:
    # depth[u] - depth[x] <= y, i.e. depth[u] <= depth[x] + y.
    # Thus, while doing a DFS of our tree we can maintain the maximum allowed depth 
    # (i.e. max(depth[x] + y)) among the events started by ancestors.
    ans = 0
    depth = [0]*(N+1)
    
    def dfs(u, current_allowed):
        nonlocal ans
        # If there is an event at u, update current_allowed using candidate allowed depth = depth[u] + y.
        for y in events_by_node[u]:
            candidate = depth[u] + y
            if candidate > current_allowed:
                current_allowed = candidate
        
        # If current allowed depth is at least the current node's depth, then u is covered.
        if depth[u] <= current_allowed:
            ans += 1
        
        # DFS update for children; note that current_allowed remains the same,
        # since an event started at an ancestor still covers deeper nodes if depth condition holds.
        for v in children[u]:
            depth[v] = depth[u] + 1
            dfs(v, current_allowed)
    
    # Start DFS from the root (person 1) with initial allowed depth set low.
    dfs(1, -1)
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()