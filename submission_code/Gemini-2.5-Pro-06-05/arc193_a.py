import sys
import heapq

def final_solve():
    """
    This function implements the solution described above.
    It reads inputs, builds the auxiliary graph using segment trees,
    and then runs Dijkstra's algorithm for each query to find the shortest path.
    """
    # Set a higher recursion limit for deep segment tree traversals
    sys.setrecursionlimit(4 * 10**5 + 5)
    input = sys.stdin.readline
    
    # --- Read Input ---
    try:
        N_str = input()
        if not N_str: return
        N = int(N_str)
        W = list(map(int, input().split()))
        intervals = []
        for i in range(N):
            L, R = map(int, input().split())
            intervals.append((L, R))
        
        Q = int(input())
        queries = []
        for _ in range(Q):
            s, t = map(int, input().split())
            queries.append((s - 1, t - 1))
    except (IOError, ValueError):
        return

    # --- Graph Construction ---
    # The coordinate space is up to 2*N
    K = 2 * N
    
    # Determine segment tree size (power of 2)
    seg_sz = 1
    while seg_sz < K:
        seg_sz *= 2
        
    num_seg_nodes = 2 * seg_sz - 1
    
    # Node indexing in the auxiliary graph
    # 0 to N-1: Interval nodes I_i
    # N to N + num_seg_nodes - 1: T_pref nodes
    # N + num_seg_nodes to N + 2*num_seg_nodes - 1: T_suff nodes
    T_pref_base = N
    T_suff_base = N + num_seg_nodes
    num_total_nodes = N + 2 * num_seg_nodes

    adj = [[] for _ in range(num_total_nodes)]

    # Build T_pref: edges from parent to children (weight 0)
    for i in range(seg_sz - 1):
        p_idx = T_pref_base + i
        c1_idx = T_pref_base + 2 * i + 1
        c2_idx = T_pref_base + 2 * i + 2
        adj[p_idx].append((c1_idx, 0))
        adj[p_idx].append((c2_idx, 0))

    # Build T_suff: edges from children to parent (weight 0)
    for i in range(seg_sz - 1):
        p_idx = T_suff_base + i
        c1_idx = T_suff_base + 2 * i + 1
        c2_idx = T_suff_base + 2 * i + 2
        adj[c1_idx].append((p_idx, 0))
        adj[c2_idx].append((p_idx, 0))

    # Connect leaf nodes to interval nodes
    for i in range(N):
        L, R = intervals[i]
        # From T_pref leaf for coord L to interval node I_i
        adj[T_pref_base + seg_sz - 1 + (L - 1)].append((i, W[i]))
        # From T_suff leaf for coord R to interval node I_i
        adj[T_suff_base + seg_sz - 1 + (R - 1)].append((i, W[i]))

    # Helper function to add edges from an interval node to a range in a segment tree
    def add_range_edges(u, tree_base, q_start, q_end):
        if q_start > q_end:
            return
        stack = [(0, 0, seg_sz - 1)]  # (node_idx_in_tree, range_l, range_r)
        while stack:
            node_idx, l, r = stack.pop()
            if r < q_start or l > q_end:
                continue
            if q_start <= l and r <= q_end:
                adj[u].append((tree_base + node_idx, 0))
                continue
            m = (l + r) // 2
            stack.append((2 * node_idx + 1, l, m))
            stack.append((2 * node_idx + 2, m + 1, r))

    # Connect interval nodes to range nodes
    for i in range(N):
        L, R = intervals[i]
        # For R_i < L_j: connect I_i to T_pref range [R, K-1] (0-based coords)
        add_range_edges(i, T_pref_base, R, K - 1)
        # For R_j < L_i: connect I_i to T_suff range [0, L-2] (0-based coords)
        add_range_edges(i, T_suff_base, 0, L - 2)

    # --- Process Queries ---
    memo_dist = {}

    for s, t in queries:
        # Use memoization to avoid re-running Dijkstra for the same source
        if s not in memo_dist:
            dist = [float('inf')] * num_total_nodes
            dist[s] = W[s]
            pq = [(W[s], s)]
            
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w in adj[u]:
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        heapq.heappush(pq, (dist[v], v))
            memo_dist[s] = dist
        
        final_dist = memo_dist[s][t]
        if final_dist == float('inf'):
            print(-1)
        else:
            print(final_dist)
            
if __name__ == "__main__":
    final_solve()