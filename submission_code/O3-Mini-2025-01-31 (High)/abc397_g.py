#!/usr/bin/env python3
import sys
from collections import deque, defaultdict

sys.setrecursionlimit(10**6)
INF = 10**9

# Global variables for the reduced graph.
# arc_list: list of (u, v) for each distinct (u,v) arc group
# cost_list: blocking cost for that arc group (i.e. number of original edges)
# graph_out: for each vertex (1-indexed) a list of (v, arc_index) for outgoing arcs.
arc_list = []      # list of (u, v)
cost_list = []     # cost (number of copies)
graph_out = []     # 1-indexed: graph_out[u] = list of (v, arc_index)

# When our DFS “blocks” some arc group we represent that by a bit in an integer mask.
# We assume the maximum number of distinct arc groups is not more than 100.
# We'll have a memo (dictionary) for the DFS state.
memo = dict()

# Given a bitmask "mask" (an integer) which represents the set of arc groups that have been "raised"
# (i.e. blocked so that arc weight = 1), we want to compute the adversary's shortest path distance from 1 to N.
# On each arc (u,v) the weight = 1 if this arc group is raised (i.e. the bit is set in mask) otherwise = 0.
# We also reconstruct one witness path (a list of arc indices along the chosen route) so that if the
# current configuration does not force cost >= d we can branch.
def compute_shortest_path(mask, N):
    # dist[v]: minimum cost from 1 to v.
    dist = [INF] * (N+1)
    # parent[v]: tuple (prev, arc_index) that was used to reach v.
    parent = [(-1, -1)] * (N+1)
    dq = deque()
    dist[1] = 0
    dq.append(1)
    while dq:
        u = dq.popleft()
        for (v, aidx) in graph_out[u]:
            # Weight on arc aidx is 1 if raised, else 0.
            w = 1 if ((mask >> aidx) & 1) else 0
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = (u, aidx)
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    # Reconstruct witness path as list of arc indices from 1 to N (if reachable)
    path = []
    if dist[N] >= INF:
        # No path exists, so consider that as INF (and witness can be empty)
        return INF, []
    # Backtrack from N to 1
    cur = N
    while cur != 1:
        par, aidx = parent[cur]
        path.append(aidx)
        cur = par
    path.reverse()
    return dist[N], path

# We now implement the DFS search.
# We try to decide, for a given target d, whether there exists a set of raised arc groups (configuration)
# with total cost not exceeding K that forces every 1-to-N path in the new graph to have at least d raised arcs.
#
# "mask" is an integer bitmask for which arc groups are raised.
# "used_cost" is the total cost (sum of cost_list for each raised group) used so far.
# "d" is our target forced cost.
def dfs_search(mask, used_cost, d, N, K):
    if used_cost > K:
        return False
    # Memoization: if we have seen this mask with a cost <= current used_cost, we can prune.
    if mask in memo and memo[mask] <= used_cost:
        return False
    memo[mask] = used_cost
    dist_val, witness_path = compute_shortest_path(mask, N)
    # If the adversary's shortest path distance is at least d, we are done.
    if dist_val >= d:
        return True
    # Otherwise, we have a witness path P with total raised cost < d.
    # (Remember: each raised arc contributes exactly 1 so the path's sum is an integer < d.)
    # We now try to add (raise) one of the arcs from the witness path that is still free.
    for aidx in witness_path:
        if not ((mask >> aidx) & 1):
            new_used = used_cost + cost_list[aidx]
            if new_used > K:
                continue
            new_mask = mask | (1 << aidx)
            if dfs_search(new_mask, new_used, d, N, K):
                return True
    return False

# For a given target d, check if there is a blocking configuration (i.e. a set of arc groups to raise)
# that forces the adversary’s 1–N least–cost (in 0/1 weights) path to have cost >= d.
def feasible(d, N, K):
    global memo
    memo = dict()
    return dfs_search(0, 0, d, N, K)

def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    it = iter(data)
    # read N, M, K from input
    N = int(next(it))
    M = int(next(it))
    K_val = int(next(it))
    # read edges
    edges = []
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        edges.append( (u, v) )
 
    # First, "condense" the graph.
    # For every directed pair (u,v) that appears, count the number of times.
    group_count = {}
    for (u, v) in edges:
        key = (u, v)
        group_count[key] = group_count.get(key, 0) + 1
 
    # Build global arc_list and cost_list.
    global arc_list, cost_list, graph_out, numArcs
    arc_list = []
    cost_list = []
    for key, cnt in group_count.items():
        arc_list.append(key)
        cost_list.append(cnt)
    numArcs = len(arc_list)
 
    # Build graph_out: for vertices 1..N, store outgoing arcs as (v, arc_index)
    graph_out = [[] for _ in range(N+1)]
    for aidx, (u, v) in enumerate(arc_list):
        graph_out[u].append( (v, aidx) )
    # (No need to sort, but you could)
 
    # Now, our new graph is defined on vertices 1..N;
    # for each arc group aidx, its weight will be 1 if raised (i.e. we choose to block all copies)
    # or 0 if left free.
    #
    # Our DFS search (dfs_search) uses a bitmask over arc groups.
    # Our goal: choose some arc groups to "raise" so that the adversary's shortest path distance (sum of raised arcs on his path)
    # is at least d. (The adversary always picks a path with as many free arcs as possible.)
    #
    # The maximum forced distance is at most (N-1) (since a simple 1-N path may have at most N-1 edges).
    ans = 0
    for d in range(N-1, -1, -1):
        if feasible(d, N, K_val):
            ans = d
            break
    sys.stdout.write(str(ans))
 
if __name__=="__main__":
    main()