import heapq
import sys

# Define infinity for distance
INF = float('inf')

# Segment Tree for Range Minimum Query and Point Update
# Stores pairs (cost, vertex_id)
# vertex_id is 1-based here, use 0-based internally in dist/W/intervals
class SegmentTree:
    def __init__(self, size):
        # size is the range of coordinates, e.g., 2N
        self.size = size # Max coordinate value
        # Let's use 1-based coordinates for segment tree nodes [1, size]
        self.tree = [(INF, -1)] * (4 * size) # Stores (min_cost, vertex_id)

    def update(self, node, start, end, idx, val):
        # Update value at index idx in the coordinate range [start, end]
        if start == end:
            # We store the minimum value encountered for this coordinate index.
            # If multiple vertices have the same endpoint coordinate, the segment tree leaf stores the min cost amongst them.
            # When a vertex's distance is updated, we update its corresponding endpoint coordinate in the segment tree.
            # The min(self.tree[node], val) ensures we keep the overall minimum for this coordinate.
            self.tree[node] = min(self.tree[node], val)
            return

        mid = (start + end) // 2
        if start <= idx <= mid:
            self.update(2 * node, start, mid, idx, val)
        else:
            self.update(2 * node + 1, mid + 1, end, idx, val)

        # Update parent node with minimum of children
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, node, start, end, l, r):
        # Query minimum value in coordinate range [l, r]
        if r < start or end < l or l > r:
            return (INF, -1)

        if l <= start and end <= r:
            return self.tree[node]

        mid = (start + end) // 2
        p1 = self.query(2 * node, start, mid, l, r)
        p2 = self.query(2 * node + 1, mid + 1, end, l, r)
        return min(p1, p2)

    # Helper function to call update from outside
    def point_update(self, idx, val):
        # idx is 1-based coordinate
        if 1 <= idx <= self.size:
             self.update(1, 1, self.size, idx, val)


    # Helper function to call query from outside
    def range_query(self, l, r):
        # l, r are 1-based coordinates
        if l > r: # Invalid range
             return (INF, -1)
        return self.query(1, 1, self.size, l, r)


def solve():
    N = int(sys.stdin.readline())
    W = list(map(int, sys.stdin.readline().split())) # 0-indexed weights W[0...N-1]
    intervals = []
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        intervals.append((l, r)) # 0-indexed list of (L, R) pairs

    Q = int(sys.stdin.readline())
    queries = []
    for _ in range(Q):
        s, t = map(int, sys.stdin.readline().split()) # 1-based vertex ids
        queries.append((s, t))

    # Max coordinate value is 2N
    max_coord = 2 * N

    results = []

    for s_node, t_node in queries:
        # Convert to 0-based index
        s_idx = s_node - 1
        t_idx = t_node - 1

        # Dijkstra initialization
        dist = [INF] * N
        dist[s_idx] = W[s_idx] # distance to reach s_node is its weight W_s
        pq = [(dist[s_idx], s_node)] # (cost, vertex_id), vertex_id is 1-based

        # Initialize segment trees
        # ST_R: Stores min(dist[v_idx] + W[v_idx], v_node) for vertices v_node with R[v_idx] = coord
        st_r = SegmentTree(max_coord)
        # ST_L: Stores min(dist[v_idx] + W[v_idx], v_node) for vertices v_node with L[v_idx] = coord
        st_l = SegmentTree(max_coord)

        # Initial update for the source vertex s_node
        # Cost stored in segment tree is dist[s_idx] + W[s_idx]
        st_r.point_update(intervals[s_idx][1], (dist[s_idx] + W[s_idx], s_node))
        st_l.point_update(intervals[s_idx][0], (dist[s_idx] + W[s_idx], s_node))

        # Use a set to keep track of visited vertices whose distance is finalized
        # visited = set() # Not strictly needed for correctness in standard Dijkstra on non-negative weights, but can optimize slightly

        found_path = False

        while pq:
            d, u_node = heapq.heappop(pq) # u_node is 1-based vertex id
            u_idx = u_node - 1 # 0-based index

            # If we found a shorter path already, ignore this one
            # Note: d is the cost to reach u_node, which is dist[u_idx]
            if d > dist[u_idx]:
                continue

            # If target is reached
            if u_node == t_node:
                found_path = True
                break

            # Find neighbors v such that R_v < L_u
            # Query ST_R for range [1, L[u_idx] - 1]
            query_r_end = intervals[u_idx][0] - 1
            if query_r_end >= 1:
                min_cost_1, v1_node = st_r.range_query(1, query_r_end)
                # min_cost_1 is min(dist[v1_idx] + W[v1_idx]) among relevant v1
                if v1_node != -1:
                    v1_idx = v1_node - 1
                    # Proposed new path cost: dist[u_idx] + W[v1_idx]
                    # dist[v1_idx] comparison uses the current minimum found for v1_idx
                    if dist[u_idx] + W[v1_idx] < dist[v1_idx]:
                        dist[v1_idx] = dist[u_idx] + W[v1_idx]
                        # Push updated vertex to PQ
                        heapq.heappush(pq, (dist[v1_idx], v1_node))
                        # Update segment trees for v1_node with its new minimum distance
                        st_r.point_update(intervals[v1_idx][1], (dist[v1_idx] + W[v1_idx], v1_node))
                        st_l.point_update(intervals[v1_idx][0], (dist[v1_idx] + W[v1_idx], v1_node))

            # Find neighbors v such that L_v > R_u
            # Query ST_L for range [R[u_idx] + 1, 2N]
            query_l_start = intervals[u_idx][1] + 1
            if query_l_start <= max_coord:
                min_cost_2, v2_node = st_l.range_query(query_l_start, max_coord)
                # min_cost_2 is min(dist[v2_idx] + W[v2_idx]) among relevant v2
                if v2_node != -1:
                    v2_idx = v2_node - 1
                    # Proposed new path cost: dist[u_idx] + W[v2_idx]
                    # dist[v2_idx] comparison uses the current minimum found for v2_idx
                    if dist[u_idx] + W[v2_idx] < dist[v2_idx]:
                        dist[v2_idx] = dist[u_idx] + W[v2_idx]
                        # Push updated vertex to PQ
                        heapq.heappush(pq, (dist[v2_idx], v2_node))
                        # Update segment trees for v2_node with its new minimum distance
                        st_r.point_update(intervals[v2_idx][1], (dist[v2_idx] + W[v2_idx], v2_node))
                        st_l.point_update(intervals[v2_idx][0], (dist[v2_idx] + W[v2_idx], v2_node))


        # After Dijkstra, dist[t_idx] holds the minimum path weight
        if dist[t_idx] == INF:
            results.append(-1)
        else:
            results.append(dist[t_idx])

    for result in results:
        sys.stdout.write(str(result) + '
')

solve()