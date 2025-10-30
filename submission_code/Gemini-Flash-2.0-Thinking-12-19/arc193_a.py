import sys
import heapq

INF_DIST = float('inf')
INF_WEIGHT = float('inf')

class SegmentTree:
    def __init__(self, max_coord):
        # max_coord is the highest possible coordinate value (e.g., 2*N)
        # Coordinates are 1-based, so range is [1, max_coord]. Size required is max_coord.
        self.coord_range_size = max_coord
        self.size = 1
        # Smallest power of 2 >= coord_range_size
        while self.size < self.coord_range_size:
            self.size *= 2

        # tree stores (min_weight, vertex_idx)
        # Internal array size 2 * self.size. Indexing: 0 to 2*size - 1.
        self.tree = [(INF_WEIGHT, -1)] * (2 * self.size)

    def _combine(self, val1, val2):
        # Combine function for minimum: takes the pair with the minimum weight
        # If weights are equal, vertex_idx doesn't matter for correctness.
        if val1[0] < val2[0]:
            return val1
        return val2

    def update(self, coord, value):
        # Update leaf node corresponding to coordinate `coord` (1-based)
        # Map 1-based coord to 0-based index for internal array leaves: coord - 1
        idx = coord - 1 + self.size
        self.tree[idx] = value
        # Propagate update up to root
        while idx > 1:
            idx //= 2
            self.tree[idx] = self._combine(self.tree[2 * idx], self.tree[2 * idx + 1])

    def query(self, l_coord, r_coord):
        # Query range of coordinates [l_coord, r_coord] (1-based, inclusive)
        # Map 1-based coords to 0-based indices for internal array range query
        
        # Handle empty or invalid range
        if l_coord > r_coord or l_coord < 1 or r_coord > self.coord_range_size:
            return (INF_WEIGHT, -1)

        # Map 1-based query range [l_coord, r_coord] to 0-based indices in the leaf array
        # The leaf corresponding to coord C (1-based) is at index self.size + C - 1
        l_idx = l_coord - 1 + self.size
        r_idx = r_coord - 1 + self.size
        
        res = (INF_WEIGHT, -1)
        
        # Standard segment tree range query on indices [l_idx, r_idx]
        # Note: range query is inclusive [l_idx, r_idx]
        while l_idx <= r_idx:
            if l_idx % 2 == 1: # l_idx is a right child in the next higher level or the current level if odd index
                res = self._combine(res, self.tree[l_idx])
                l_idx += 1
            if r_idx % 2 == 0: # r_idx is a left child in the next higher level or the current level if even index
                res = self._combine(res, self.tree[r_idx])
                r_idx -= 1
            l_idx //= 2
            r_idx //= 2
            
        return res

def solve():
    # Use faster I/O
    input = sys.stdin.readline

    N = int(input())
    W = [0] + list(map(int, input().split())) # 1-indexed
    intervals = [(0, 0)] # 1-indexed
    for _ in range(N):
        l, r = map(int, input().split())
        intervals.append((l, r))

    MAX_COORD = 2 * N

    Q = int(input())
    
    # Precompute initial minimum weight vertices at each coordinate for L and R
    # This helps in faster segment tree initialization per query.
    # Store lists of (weight, vertex_idx) for vertices whose L or R is at a coordinate.
    # Use lists to handle multiple vertices per coordinate.
    vertices_by_l = [[] for _ in range(MAX_COORD + 1)]
    vertices_by_r = [[] for _ in range(MAX_COORD + 1)]

    for i in range(1, N + 1):
        vertices_by_l[intervals[i][0]].append((W[i], i))
        vertices_by_r[intervals[i][1]].append((W[i], i))

    # Sort lists by weight to easily find the minimum at each coordinate
    for coord in range(1, MAX_COORD + 1):
        if vertices_by_l[coord]:
             vertices_by_l[coord].sort()
        if vertices_by_r[coord]:
             vertices_by_r[coord].sort()

    for _ in range(Q):
        s, t = map(int, input().split())

        dist = [INF_DIST] * (N + 1)
        dist[s] = W[s]
        
        pq = [(dist[s], s)]
        # heapq.heapify(pq) # Not strictly needed when adding one by one initially

        visited = [False] * (N + 1)

        # Initialize segment trees for this query
        current_st_l = SegmentTree(MAX_COORD)
        current_st_r = SegmentTree(MAX_COORD)

        # Build segment trees from precomputed sorted lists of vertices at each coordinate
        # For each coordinate, the initial value in the ST leaf is the minimum weight vertex
        # at that coordinate.
        # This build is O(N + MAX_COORD) for leaves if min is precomputed or O(N log N) if sorting lists here.
        # Propagation is O(MAX_COORD). Total O(N log N) or O(N) per build depending on precomputation strategy.
        # We precomputed lists and sorted them, so finding min for leaf is O(1).
        # Setting leaves O(MAX_COORD), propagating O(MAX_COORD). Total O(N) build.

        for coord in range(1, MAX_COORD + 1):
             if vertices_by_l[coord]:
                current_st_l.tree[current_st_l.size + coord - 1] = vertices_by_l[coord][0]
             if vertices_by_r[coord]:
                current_st_r.tree[current_st_r.size + coord - 1] = vertices_by_r[coord][0]

        # Propagate min up the tree
        for i in range(current_st_l.size - 1, 0, -1):
             current_st_l.tree[i] = current_st_l._combine(current_st_l.tree[2 * i], current_st_l.tree[2 * i + 1])
             current_st_r.tree[i] = current_st_r._combine(current_st_r.tree[2 * i], current_st_r.tree[2 * i + 1])

        while pq:
            cost, u = heapq.heappop(pq)

            if cost > dist[u]:
                continue
            if visited[u]:
                continue
            visited[u] = True

            # Vertex u is processed. Remove its contribution from segment trees.
            # Set its entry at its coordinates to INF.
            # If u was the minimum at its coordinate, this correctly removes the minimum.
            # If u was not the minimum, this update might be wrong. However, the `visited[v]` check during query handles this.
            # When a vertex is visited, it's out. Update its entries to INF.
            # The `query` method will return the minimum *among all vertices* in the range, including visited ones potentially.
            # The `while True` loop logic explicitly checks `visited[v]` and skips if true, then removes the visited vertex from ST for good.

            # Set vertex u's weight to INF at its coordinate leaves in STs
            # This works because the repeated query loop checks visited status.
            current_st_l.update(intervals[u][0], (INF_WEIGHT, u))
            current_st_r.update(intervals[u][1], (INF_WEIGHT, u))

            if u == t:
                break

            # Find neighbors v where R_u < L_v. Query current_st_l for range [R_u + 1, MAX_COORD].
            r_u = intervals[u][1]
            query_range_l_coords = [r_u + 1, MAX_COORD]
            if query_range_l_coords[0] <= query_range_l_coords[1]:
                while True:
                    min_w_v, v = current_st_l.query(query_range_l_coords[0], query_range_l_coords[1])

                    if v == -1 or min_w_v == INF_WEIGHT:
                         break # No more potentially unvisited vertices in this range

                    # If the vertex found is visited, remove it from ST for this run and find the next one
                    if visited[v]:
                        # Remove this visited vertex from the ST so it's not returned by future queries
                        current_st_l.update(intervals[v][0], (INF_WEIGHT, v))
                        continue # Query again for the next minimum in the range

                    # Vertex v is unvisited. Process it.
                    new_dist_v = dist[u] + min_w_v

                    if new_dist_v < dist[v]:
                        dist[v] = new_dist_v
                        heapq.heappush(pq, (dist[v], v))

                    # Remove v from ST_L permanently for this run as it's been found via L_v query
                    # This ensures each vertex is found via ST_L query at most once.
                    current_st_l.update(intervals[v][0], (INF_WEIGHT, v))
                    # Continue the while loop to find the next best neighbor in this range

            # Find neighbors v where R_v < L_u. Query current_st_r for range [1, L_u - 1].
            l_u = intervals[u][0]
            query_range_r_coords = [1, l_u - 1]
            if query_range_r_coords[0] <= query_range_r_coords[1]:
                 while True:
                    min_w_v, v = current_st_r.query(query_range_r_coords[0], query_range_r_coords[1])

                    if v == -1 or min_w_v == INF_WEIGHT:
                         break # No more potentially unvisited vertices in this range

                    # If the vertex found is visited, remove it from ST for this run and find the next one
                    if visited[v]:
                        current_st_r.update(intervals[v][1], (INF_WEIGHT, v))
                        continue # Query again for the next minimum in the range

                    # Vertex v is unvisited. Process it.
                    new_dist_v = dist[u] + min_w_v

                    if new_dist_v < dist[v]:
                        dist[v] = new_dist_v
                        heapq.heappush(pq, (dist[v], v))

                    # Remove v from ST_R permanently for this run as it's been found via R_v query
                    # This ensures each vertex is found via ST_R query at most once.
                    current_st_r.update(intervals[v][1], (INF_WEIGHT, v))
                    # Continue the while loop to find the next best neighbor in this range
                

        if dist[t] == INF_DIST:
            print(-1)
        else:
            print(dist[t])

# Set recursion depth limit if necessary (though iterative segment tree shouldn't need deep recursion)
# sys.setrecursionlimit(300000) 
solve()