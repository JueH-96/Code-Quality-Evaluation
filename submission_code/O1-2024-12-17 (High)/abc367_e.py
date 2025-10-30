def main():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque
    from math import log2, ceil
    input_data = sys.stdin.read().strip().split()
    
    # Read inputs
    N = int(input_data[0])
    K = int(input_data[1])
    
    X = list(map(int, input_data[2:2+N]))
    A = list(map(int, input_data[2+N:2+2*N]))
    
    # If no operations, simply print A
    if K == 0:
        print(*A)
        return
    
    # Build the function p: i -> X[i] - 1 (0-based)
    p = [x - 1 for x in X]
    
    # Build reverse-graph adjacency (rev[u] = list of nodes w s.t. p[w] = u)
    rev = [[] for _ in range(N)]
    for i in range(N):
        rev[p[i]].append(i)
    
    # 1) Find which nodes are truly in a cycle via "in-degree = 0" elimination
    inDeg = [0]*N
    for i in range(N):
        inDeg[p[i]] += 1
    
    inCycle = [True]*N
    queue = deque()
    for i in range(N):
        if inDeg[i] == 0:
            inCycle[i] = False
            queue.append(i)
    
    while queue:
        v = queue.popleft()
        # For each w that points to v
        for w in rev[v]:
            inDeg[w] -= 1
            if inDeg[w] == 0 and inCycle[w]:
                inCycle[w] = False
                queue.append(w)
    
    # 2) Identify the actual cycles and assign IDs/positions
    cycID = [-1]*N   # which cycle a node belongs to
    cycPos = [-1]*N  # index within that cycle
    cycles = []      # list of cycles (each cycle is a list of nodes)
    
    def find_cycle(start, cid):
        # Gather the cycle starting from "start"
        cycle_nodes = []
        # First pass: mark nodes with -2 until we revisit or reach known cycle
        v = start
        while cycID[v] == -1:
            cycID[v] = -2    # temporary mark
            v = p[v]
        # If cycID[v] == -2, we've found a new cycle
        if cycID[v] == -2:
            # Reconstruct the cycle
            cycle_start = v
            cycle_nodes.append(v)
            cycID[v] = cid
            cycPos[v] = 0
            w = p[v]
            idx = 1
            while w != cycle_start:
                cycle_nodes.append(w)
                cycID[w] = cid
                cycPos[w] = idx
                idx += 1
                w = p[w]
            cycles.append(cycle_nodes)
            return True
        return False
    
    cID = 0
    for i in range(N):
        if inCycle[i] and cycID[i] == -1:
            # find a new cycle
            got_cycle = find_cycle(i, cID)
            if got_cycle:
                cID += 1
    
    # 3) BFS in reverse to compute dist[] (distance to the cycle) and cycNode[] (the cycle-node reached)
    dist = [-1]*N
    cycNode = [-1]*N
    q = deque()
    
    # Initialize queue with cycle nodes
    for i in range(N):
        if cycID[i] >= 0:  # i is on a cycle
            dist[i] = 0
            cycNode[i] = i
            q.append(i)
    
    # BFS in reverse
    while q:
        u = q.popleft()
        for w in rev[u]:
            if dist[w] == -1:  # not visited yet
                dist[w] = dist[u] + 1
                cycNode[w] = cycNode[u]
                q.append(w)
    
    # 4) For each cycle, store the length
    cycLenArr = [0]*cID
    for i in range(cID):
        cycLenArr[i] = len(cycles[i])
    
    # Prepare arrays so every node has cycID, cycPos, cycLen in O(1)
    # (if a node is not a cycle node, it inherits from cycNode[node])
    # cycID[i], cycPos[i] were set for cycle nodes; for others they remain -1
    for i in range(N):
        if cycID[i] >= 0:
            # already known
            pass
        else:
            # use the cycle information from cycNode[i]
            root = cycNode[i]
            cycID[i] = cycID[root]
            cycPos[i] = cycPos[root]
    # Build a quick cycLen for each node as well
    nodeCycLen = [0]*N
    for i in range(N):
        nodeCycLen[i] = cycLenArr[cycID[i]] if cycID[i] >= 0 else 0
    
    # 5) Build sparse table (upDist) for jumping up to dist[i] steps (<= N)
    #    We only need log2(N) not log2(K), because we'd never jump more than dist[i]
    #    steps outside the cycle. Once on the cycle, we do modulo for large K.
    from math import floor, log2
    
    maxH = int(log2(N)) + 2  # a bit of cushion
    # upDist[h][i] = p^(2^h)(i)
    upDist = [ [0]*N for _ in range(maxH) ]
    
    for i in range(N):
        upDist[0][i] = p[i]
    
    for h in range(1, maxH):
        for i in range(N):
            upDist[h][i] = upDist[h-1][ upDist[h-1][i] ]
    
    def forward_jump(start, steps):
        """Return the node reached from 'start' after 'steps' applications, 
           where steps <= N (we only use it if steps < dist[start] or dist[start] itself)."""
        node = start
        h = 0
        for h in reversed(range(maxH)):
            step_size = 1 << h
            if step_size <= steps:
                node = upDist[h][node]
                steps -= step_size
        return node
    
    # 6) Compute final array
    res = [0]*N
    for i in range(N):
        d = dist[i]
        if K < d:
            # We'll jump exactly K steps
            finalPos = forward_jump(i, K)
        else:
            # Jump d steps first (puts us onto the cycle)
            # then do (K - d) mod cycLen
            cid = cycID[i]
            pos = cycPos[i]
            length = nodeCycLen[i]  # length of the cycle
            steps_in_cycle = (K - d) % length
            finalPos = cycles[cid][ (pos + steps_in_cycle) % length ]
        res[i] = A[finalPos]
    
    print(*res)

# Don't forget to call main()
if __name__ == "__main__":
    main()