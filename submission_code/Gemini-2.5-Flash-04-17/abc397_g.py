import sys
from collections import deque

# Function to check if it's possible to achieve a shortest path of at least target_dist
# This is equivalent to checking if the minimum number of original edges that must be assigned weight 1
# to ensure that all paths from 1 to N have length >= target_dist is <= K.
# This minimum number is equivalent to the minimum cost S-T cut in a related flow network,
# which can be found by computing the shortest path in a cost graph.
# Cost graph nodes (u, w) where u is original vertex, w is accumulated weight < target_dist.
# The cost of an edge (u, w) -> (v, w) (representing original edge u->v assigned weight 0) is 0.
# The cost of an edge (u, w) -> (v, w+1) (representing original edge u->v assigned weight 1) is 1.
# dist[u][w] = min cost (number of original edges assigned weight 1) to reach state (u, w).
def can(target_dist, N, K, edges):
    if target_dist == 0:
        return True # Shortest path is always >= 0

    # State: (vertex u, accumulated weight w)
    # accumulated weight w ranges from 0 to target_dist - 1
    # dist[u][w] = min number of original edges assigned weight 1 to reach vertex u with accumulated weight w
    dist = [[float('inf')] * target_dist for _ in range(N + 1)]
    
    # Use 0-1 BFS (deque) since edge costs are 0 or 1
    q = deque()

    # Starting state: vertex 1, accumulated weight 0
    dist[1][0] = 0
    q.append((1, 0))

    while q:
        u, w = q.popleft()

        # If we already found a better path to this state, skip
        if dist[u][w] == float('inf'):
            continue

        # Explore neighbors using original edges
        for u_orig, v_orig in edges:
            if u_orig == u:
                v = v_orig
                
                # Option 1: Assign original edge (u, v) weight 0
                # Transition from state (u, w) to state (v, w) with cost 0
                next_w = w
                cost = dist[u][w]
                if next_w < target_dist:
                    if cost < dist[v][next_w]:
                        dist[v][next_w] = cost
                        q.appendleft((v, next_w)) # Add to front for cost 0

                # Option 2: Assign original edge (u, v) weight 1
                # Transition from state (u, w) to state (v, w + 1) with cost 1
                next_w = w + 1
                cost = dist[u][w] + 1
                if next_w < target_dist:
                    if cost < dist[v][next_w]:
                        dist[v][next_w] = cost
                        q.append((v, next_w)) # Add to back for cost 1
    
    # The minimum number of weight 1 edges needed to have a path from 1 to N with length < target_dist
    # is the minimum cost to reach any state (N, w) where w < target_dist.
    min_weight1_needed = float('inf')
    for w in range(target_dist):
        min_weight1_needed = min(min_weight1_needed, dist[N][w])

    # If this minimum number of required weight 1 edges is > K,
    # it means we cannot create any path with length < target_dist using at most K weight 1 edges.
    # Therefore, the shortest path MUST be >= target_dist.
    # Note: If min_weight1_needed is infinity, it means no path with length < target_dist exists in the
    # original graph even if we could choose weights freely. This would imply shortest path is >= target_dist.
    # Since N is reachable from 1 in the original graph, a path with length > 0 always exists unless N=1 (not allowed).
    # The only case where min_weight1_needed could be infinity for D > 0 is if vertex N is unreachable from 1
    # while staying below target_dist accumulated weight, even with optimal weight 1 edge placement.
    # If min_weight1_needed is infinity, it is definitely > K.
    return min_weight1_needed > K

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        edges.append((u, v))
    
    # Binary search for the maximum possible shortest distance D
    # The shortest distance can range from 0 up to N-1 (maximum simple path length)
    # or up to K (maximum number of weight 1 edges we can choose).
    # A safe upper bound for the maximum shortest distance is N.
    # Search space for D is [0, N]. We search on [low, high), initially [0, N+1].
    low = 0
    high = N + 1 # Maximum possible shortest path can be at most N-1.

    ans = 0
    while high - low > 1:
        mid = (low + high) // 2
        if can(mid, N, K, edges):
            ans = mid
            low = mid
        else:
            high = mid
            
    print(ans)

if __name__ == "__main__":
    main()