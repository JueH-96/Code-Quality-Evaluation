import sys
import heapq
import collections

def solve():
    """
    This function encapsulates the entire solution logic.
    """
    try:
        line = sys.stdin.readline()
        if not line: return
        N, M, X1 = map(int, line.split())
        
        trains_data = []
        for _ in range(M):
            trains_data.append(list(map(int, sys.stdin.readline().split())))
    except (IOError, ValueError):
        return

    # Group trains by arrival and departure cities (0-indexed)
    arrivals_by_city = collections.defaultdict(list)
    departures_by_city = collections.defaultdict(list)
    for i in range(M):
        a, b, s, t = trains_data[i]
        arrivals_by_city[b - 1].append((t, i))
        departures_by_city[a - 1].append((s, i))

    # Sort arrivals by time for each city
    sorted_arrivals = {}
    for c in range(N):
        if c in arrivals_by_city:
            arrivals_by_city[c].sort()
            sorted_arrivals[c] = arrivals_by_city[c]

    # --- Graph Construction ---
    # Nodes 0 to M-1 represent trains.
    # Nodes M onwards are auxiliary nodes for city arrivals.
    
    aux_node_start_idx = {}
    num_aux_nodes = 0
    cur_aux_idx = M
    
    for c in range(N):
        if c in sorted_arrivals:
            aux_node_start_idx[c] = cur_aux_idx
            num_aux_nodes += len(sorted_arrivals[c])
            cur_aux_idx += len(sorted_arrivals[c])
            
    num_total_nodes = M + num_aux_nodes
    adj = [[] for _ in range(num_total_nodes)]

    # Add edges related to arrivals
    for c in sorted_arrivals:
        start_idx = aux_node_start_idx[c]
        arrivals = sorted_arrivals[c]
        for k in range(len(arrivals)):
            t, train_idx = arrivals[k]
            aux_node_idx = start_idx + k
            
            # Edge from train node to its corresponding arrival aux node
            adj[train_idx].append((aux_node_idx, t))
            
            # Edge from previous aux node to current one to propagate the max
            if k > 0:
                adj[aux_node_idx - 1].append((aux_node_idx, 0))

    # Add edges related to departures
    from bisect import bisect_right
    for c in departures_by_city:
        if c not in sorted_arrivals:
            continue
            
        arrivals = sorted_arrivals[c]
        arr_times = [t for t, i in arrivals]
        start_idx = aux_node_start_idx[c]
        
        for s, train_idx in departures_by_city[c]:
            # Find the last arrival with time T <= S
            k = bisect_right(arr_times, s)
            if k == 0:
                continue
            
            # The dependency is on the max arrival time up to the (k-1)-th arrival
            aux_node_idx = start_idx + k - 1
            
            # Edge from aux node to the departing train node
            adj[aux_node_idx].append((train_idx, -s))

    # --- Longest Path Calculation ---
    dist = [-1] * num_total_nodes
    
    # Initialize distances for train nodes. X_i >= 0.
    for i in range(M):
        dist[i] = 0
    dist[0] = X1
    
    # Max-priority queue, stores (-distance, node_index) for Dijkstra
    pq = [(-X1, 0)]

    while pq:
        d, u = heapq.heappop(pq)
        d = -d
        
        if d < dist[u]:
            continue
            
        for v, w in adj[u]:
            if dist[u] + w > dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (-dist[v], v))
                
    # The results are the calculated delays for trains 2 to M
    result = [dist[i] for i in range(1, M)]
    print(*result)

solve()