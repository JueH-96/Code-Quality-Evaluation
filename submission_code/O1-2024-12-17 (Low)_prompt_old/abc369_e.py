def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # We'll parse the input first.
    idx = 0
    N = int(input_data[idx]); idx += 1
    M = int(input_data[idx]); idx += 1

    # Store the bridges in an array.
    # Each bridge has (u, v, t).
    # Note that 1 <= U_i < V_i <= N, so no need to worry about zero-based indexing issues for the islands.
    # We'll keep them 1-based for convenience.
    bridges = []
    for _ in range(M):
        u = int(input_data[idx]); idx += 1
        v = int(input_data[idx]); idx += 1
        t = int(input_data[idx]); idx += 1
        bridges.append((u, v, t))

    Q = int(input_data[idx]); idx += 1

    # Observations and approach:
    #
    # 1) N up to 400, M up to 200,000. We have to be able to handle up to 3000 queries, each query requiring
    #    using K_i given edges (where K_i <= 5).
    # 2) Because N is only 400, we can consider computing all-pairs shortest paths. However, M can be large,
    #    which suggests we can't just store a direct adjacency of size N*N with naive Floyd-Warshall, as that
    #    might be too large in time (O(N^3) = 64 million ops is borderline feasible in optimized code, but fine with fast languages).
    #    But we need more than "just" standard shortest paths, because we must use some edges at least once.
    #
    # 3) Because we must use certain edges (K <= 5) at least once, we can think of the problem as visiting a "set" of edges
    #    in a path from 1 to N, crossing each required edge at least once in either direction. 
    #    This is reminiscent of the "bitmask on edges" approach, but that typically ends up with huge complexity if we revolve around M edges.
    #    But K <= 5, so we only care about those K edges in terms of visitation constraints. 
    #
    # 4) We can treat each required edge as an entity that must be "visited." Because K <= 5, we can handle the subsets of these edges with bitmask DP.
    #    We also need to handle the possibility that we can cross the same edge multiple times in different directions.
    #
    # 5) We should set up a scenario:
    #        We'll have a graph with adjacency list. We'll compute standard shortest path distances between all vertices. 
    #        Because N up to 400, we can indeed run Floyd Warshall in O(N^3)=64,000,000 (64 million) which might be borderline but possibly still feasible in C++.
    #        In Python, that might be too slow. Alternatively, we can do repeated D'Johnson or repeated D' for each node we might need in the DP, but that might also be quite large if we do it incorrectly. 
    #
    #    But note, for each query, we only need to do a DP over the states where we track which of the K edges have been used. The states for the DP will revolve around the *vertices* and the bitmask of used edges. 
    #    That means we would need distances between all pairs of vertices in order to quickly handle transitions. 
    #    The reason is: from any vertex, we can jump to any other vertex in cost = dist(u,v). Then we might choose to use one (or more) of the required edges in that jump. But we can't just "jump" for free if we want to use a specific edge. We need to carefully incorporate the usage of edges if we explicitly decide to traverse them. 
    #
    # 6) Another approach: Because K <= 5, we can treat each of the K edges as a "special object" we need to incorporate at least once in our route from 1 to N. 
    #    We can do a TSP-like approach in a graph not on the vertices, but on the endpoints of these edges plus 1 and N. However, an edge can be used in either direction, so we could incorporate an approach of enumerating sequences that cover all required edges. 
    #
    # 7) More direct DP approach:
    #    Let qEdges = the set of K edges we must use. Each edge e_i has endpoints (a_i, b_i). We can store the distance between any pair of endpoints a and b using a standard distance matrix dist[a][b] (the shortest path ignoring the usage constraints, just the minimal time). We'll compute this once for the entire graph using e.g. Floyd-Warshall or repeated D' (since N=400, Floyd-Warshall might be borderline but maybe doable in optimized Python).
    #
    #    Then, we create a DP with states:
    #       dp[v][mask], meaning the minimal time to start from island 1, use the set of required edges corresponding to 'mask' (in some order), and end at vertex v. Once we've used all edges in the mask, we are in that "used" state. 
    #    But how do we incorporate "using an edge e_i"? We can "activate" usage of e_i when we cross from a_i to b_i or from b_i to a_i in a single jump. So we must track that we specifically took that path in a single jump. But in standard usage, we might jump from a to b in a path that doesn't necessarily match exactly that single edge as a direct bridging. Actually, to "use" a required edge, we specifically have to cross that edge. That suggests we can't just jump using the entire shortest path. We must break it so that crossing a required edge is recognized as an event.
    #
    # 8) We can store a compressed graph that has as vertices: {1, N} plus both endpoints of each of the K edges. Potentially up to K * 2 + 2 distinct vertices. We can compute the shortest path between each pair of these vertices in the original graph ignoring usage constraints. Then from that compressed graph, we do a TSP-like DP on that set of up to 2*K+2 nodes, but with the requirement that each of the K edges is used at least once. 
    #
    #    But how do we ensure we "use" an edge e_i? Using e_i means traveling from its endpoint a_i to b_i directly in the original graph with a cost T_i (the time for that single edge). We do want to be allowed to combine traveling from a_i to b_i in a route that might pass through other nodes or edges in the original graph, but that wouldn't count as "using" the required edge if we didn't specifically cross e_i. 
    #    So we must provide the possibility that from a_i to b_i, we either do "not use e_i" in cost = dist[a_i][b_i], or we do "use e_i" in cost = T_i. If we do "use e_i," that toggles the usage bit for that edge. 
    #
    # 9) The maximum K=5, meaning we can have up to 2^5 = 32 states in terms of used edges. We'll do a DP over the compressed set of up to 2*K+2 nodes. Actually, the number of relevant "node states" in the DP is up to 2*K+2 because we might consider each endpoint plus 1, plus N. Actually, 1 or N could coincide with some endpoints, but let's not overcomplicate. 
    #
    #    But the transitions are tricky because from a node x to a node y in the compressed set, we have two possible ways to travel:
    #      - We do not use any of the K edges. Then cost is dist[x][y] if we do not cross any of the K edges. But wait, if in the path from x to y in the original graph's shortest path we happen to cross e_i, that doesn't "count" as using e_i" if we want to specifically cross that physical edge? This is ambiguous from the problem statement, but it states "Find the minimum time required to travel from island 1 to island N using each of these bridges at least once. Only consider the time spent crossing bridges. You can cross the given bridges in any order and in any direction." 
    #      Actually, it means that crossing the required edge physically in the route is what matters. If the original shortest path from x to y happens to physically traverse that edge, that does count. But we have no easy way to see if that path uses that edge if we just store the standard "dist[x][y]." Because the standard shortest path might use a different edge. 
    #
    #    Hence, we can't just rely on the standard shortest path ignoring usage constraints. We need to explicitly incorporate the usage of the K edges. 
    #
    # 10) Another approach:
    #    Because K <= 5, we can treat each required edge as a special link. In a route that uses all required edges, each required edge is crossed at least once. We can cross other edges any number of times. 
    #    A standard approach for "visit a subset of edges" is to do a route that visits a set of "waypoints" on the graph. But those "waypoints" are edges, not vertices. 
    #
    #    We can do a DP on states: (v, mask), where v is a vertex in [1..N], mask is which of the K edges are used. Then we have transitions from (v, mask) to (v', mask') if we travel from v to v' using some path in the original graph. The cost is the minimal time crossing the edges in that path. Along that path, if we cross any of the K edges e_i, we set the i-th bit in mask'. So mask' = mask OR some combination. 
    #
    #    But enumerating all possible intermediate v' from v is huge if we consider M up to 200k. However, we can store for each edge in the entire graph, we can go from v to either endpoint for that edge. That is a standard adjacency approach. But we have to update mask if that edge is in our required set. 
    #
    #    Then we could run a D' (like a BFS or D' with a priority queue) over the extended state space of size N * 2^K = 400 * 32 = 12800 states, which is not large. The adjacency in the state space is each edge (u->v) in the original graph leading from (u,mask) to (v, mask') where mask' is mask plus the bit for that edge if it's a required edge. The cost is T(edge). Then we do a standard shortest path in that expanded state graph from (1,0) to (N, (1<<K)-1). That is feasible. 
    #
    #    The only challenge is that M can be 200k, meaning building adjacency lists is big but feasible in memory. Then running a D' for each query would be too large (Q up to 3000, M up to 200k => 600 million edges across queries is too big). 
    #
    #    We can do that BFS for each query? That is too big. We need an approach that reuses precomputed info across queries. 
    #
    # 11) We can build the "super graph" with N * 2^K states for the entire set of edges that might be required. But each query has a different subset of K edges (K up to 5) => we cannot just fix one set of K edges for all queries. 
    #    That suggests a separate BFS for each query is not efficient. 
    #
    # 12) Because Q can be up to 3000, but K_i up to 5, we suspect the approach for each query is some DP with 2^K states. But we must in some way quickly incorporate the cost of traveling between states. 
    #
    # 13) Another angle: We note that for a given query with K edges, the usage constraint is that each of these K edges is crossed at least once. Let those edges be e_1, e_2, ..., e_K with endpoints (a_i, b_i) and cost t_i. We can cross any edges we want any number of times. 
    #
    #    The minimal path that ensures usage of those K edges might be done in an order that is interleaved with traveling from 1 to N. We can do something like: 
    #      - Start at 1
    #      - Possibly move around for some time
    #      - Cross the 1st required edge (somewhere in the route)
    #      - Possibly move around more
    #      - Cross the 2nd required edge, etc.
    #      - End at N
    #
    #    Each time we cross a required edge e_i, we pay t_i for that crossing. The rest of the traveling can use any edges (not necessarily required) but that cost is the sum of the T_i for the edges used. We want to do it in a minimal sum of times. 
    #
    #    But we only have up to 5 required edges. A standard trick is: we can treat each required edge crossing as an event that "transitions" from some vertex to another vertex with a known cost. But we also need the possibility to travel among these vertices ignoring which edges are used if they are not required. 
    #
    #    If we precompute the shortest path cost among all pairs of vertices in the graph ignoring usage constraints, let that be dist[u][v]. Then crossing a required edge e_i from a_i to b_i has cost t_i, but we also might do a side trip to get from current vertex x to a_i, cross the edge e_i to b_i, and continue. So that cost would be dist[x][a_i] + t_i. Then from b_i we can continue onward. Or we could cross it in the reverse direction with cost dist[x][b_i] + t_i, ending up at a_i. 
    #
    #    So for each required edge, we have two "possible transitions":
    #      crossing e_i forward: from x to b_i at cost dist[x][a_i] + t_i,
    #      crossing e_i backward: from x to a_i at cost dist[x][b_i] + t_i.
    #
    #    Then we can do a bitmask DP on the usage of edges. We'll define dp[v][mask] = minimal cost to be at vertex v having used all edges in mask. Then we do transitions:
    #      1) Move from v to w using the minimal path cost dist[v][w] (which does not use any required edge, or it may use them but that doesn't "count" as fulfilling the usage requirement unless we specifically cross that required edge in a "direct" way?). Actually, wait, if in the path dist[v][w], we physically cross a required edge e_i, that should count. But we have no direct handle on that in dist[v][w], because that path might skip or might not skip that edge in the best route. 
    #
    #    The problem is that "using" an edge means physically crossing that specific edge, not just traveling between its endpoints. 
    #
    # 14) We might do a multi-graph approach: For each of the M edges, if it's not in the set of required edges, it can be used any time we want in normal adjacency. If it is one of the required edges, we can have two adjacency edges for it in the state space: one that doesn't "use" it, with cost T_i if there's a path that avoids physically crossing it? That wouldn't make sense. 
    #
    # 15) Perhaps the simpler approach is to run a shortest path in the "state graph" for each query. The "state graph" has N*(2^K) states. For each original edge (u,v,t), from (u, mask) we can go to (v, mask') with cost t, where mask' = mask OR bit_i if that edge is the i-th required edge. If it isn't a required edge, mask' = mask. Then from (v, mask) we can go to (u, mask') similarly, because the edges are bidirectional. Then we run D' from (1, 0) to (N, (1<<K)-1). That gives the minimal cost. 
    #
    #    Implementation detail: We'll do this for each query. We have M up to 200k and Q up to 3000, meaning 600 million edges if we do a naive approach for each query => too large. We must optimize. 
    #
    # 16) Notice that K_i can vary per query. But K_i <= 5. We can have many queries with different sets of required edges. We can't pre-build a single state graph for all edges because which edges are "required" changes the bit assignment. 
    #
    # 17) But K_i up to 5, B_{i,1},..., B_{i,K_i} are distinct. We can map those K edges to bits 0..K-1. Then in the adjacency, each edge e_j in the entire set M is either "not required" or "it is the r-th required edge." Then we do a standard D' in the state graph. The cost of building that adjacency on the fly is large if we do it naively (200k edges each time). But we can do a faster approach if we store the adjacency in a compressed form. 
    #
    #    We can store for each node v a list of edges (v-> nextNode, cost, index of the edge). Then, for each query, we quickly identify which edges of those adjacency are among the K required edges. We create a small lookup table for "edge_index -> bitmask" that is 0 if not required, or (1<< help) if it is the i-th required edge. Then we do a normal D' in a graph with N*(2^K) states and up to M*(2) transitions (since each edge is bidirectional). 
    #    The time per query for a normal D' on N*(2^K) ~ 400*32=12800 states, each state with adjacency degree up to deg(v) transitions. Summed over all v, adjacency is 2*M=400k in worst case. 12800 * log(12800) is ~130k. Then multiplied by 200k edges is 2.6e10 => too big. 
    #
    # 18) We need a more efficient method. If K_i is small, we might attempt a meet-in-the-middle approach:
    #    We want a path from 1 to N that crosses K edges. We might reorder them in any sequence. The cost to "connect" crossing e_i to crossing e_j can be computed if we know the minimal cost route from an endpoint of e_i to an endpoint of e_j that does not necessarily cross e_j. But we do care if that route crosses some required edges? Actually, we only care about the cost in time. If it crosses required edges that we've not "officially used" yet, that might be beneficial but complicated to track. 
    #
    # 19) Because K <= 5, an alternative is to consider all permutations of the K edges, e_1 ... e_K. Then for each permutation, we have 2^K ways to choose directions (forward or backward crossing for each required edge) => each edge can be crossed in 2 ways (a_i->b_i or b_i->a_i). So total permutations = K! <= 120, total direction sets = 2^K <= 32, combined is at most 3840. Then for each arrangement, we compute the cost as:
    #      cost(1 -> startOf e_1) + t(e_1) + cost(endOf e_1 -> startOf e_2) + t(e_2) + ... + cost(endOf e_K -> N),
    #    where cost(x->y) is the minimal path ignoring usage constraints. We'll compute dist[][] with a standard shortest path (like Floyd-Warshall or repeated D') once. We just pick dist[x][y] from that matrix. 
    #    Then we take the minimal among all permutations/direction combinations. 
    #    This works because we know we must cross each required edge exactly once? Actually, we must cross each at least once, so crossing them more than once can't be cheaper, because crossing an edge has a nonnegative cost. So crossing each exactly once suffices for minimal cost if the cost is strictly positive. 
    #
    #    Indeed, there's no advantage to crossing a required edge more than once, because each crossing costs T_i > 0. 
    #
    #    Therefore, the best route will cross each required edge exactly once in either direction, because crossing multiple times only adds cost. 
    #
    #    So that drastically simplifies the problem. We only need the minimal distances between relevant vertices: 1, N, and the endpoints of each bridge. For each edge e_i = (u_i, v_i), we define e_i's endpoints as u_i, v_i. Then in the route, if we choose forward crossing, we do "cost(... -> u_i) + T_i + cost(v_i -> ...)", or if backward crossing, "cost(... -> v_i) + T_i + cost(u_i -> ...)". 
    #
    #    So to handle each query:
    #      - Let the set of required edges be e_1,...,e_K. Gather their endpoints: (u_i,v_i).
    #      - We compute all pairs dist among the set of nodes S = {1, N, u_1, v_1, ..., u_K, v_K} using the large dist[][] matrix that is available from the precomputation. 
    #      - Then we try all permutations of the K edges in some order, and for each edge choose forward or backward crossing. The cost is:
    #          startCost = dist(1, chosenEndpointOf e_1) + T(e_1) + ...
    #          + sum of dist(endOf e_i, startOf e_{i+1}) + T(e_{i+1})
    #          + dist(endOf e_K, N)
    #
    #      - We pick the minimal cost. 
    #
    # 20) This is likely feasible if we can precompute the dist[][] for all N up to 400 using an all-pairs approach. O(N^3) = 64 million might be borderline in Python, but could be tried with an efficient approach (like using adjacency and repeated D' from each node => O(N (M log N)), which is 400*(200k log 400) ~ 400*(200k*9) = 400*1.8e6=7.2e8 => that is large. 
    #    Floyd Warshall is 64 million steps, which might be done with some optimization in Python. It's borderline but might pass if carefully optimized with fast IO. 
    #
    # 21) Then for each query, we do the permutations. K! <= 120, plus 2^K <= 32 => 3840 combinations. That times Q up to 3000 => about 11.5 million checks, which is probably too big in Python. We can optimize if K_i < 5. But K_i can be up to 5. 3,000 * 3,840 = 11,520,000 => borderline but maybe with efficient code it could pass. 
    #
    # 22) We can reduce the factor if we do a bitset dynamic approach for the permutation/direction part. Actually, we can do a TSP approach over K edges in O(K * 2^K), which is at most 5*32=160 operations to get the minimal path if we incorporate the direction choices cleverly. Then multiply that by Q=3000 => 480,000 => quite feasible. 
    #
    #    We'll define 2K distinct "points": For each edge i, we have 2 possible ways to cross: "start at u_i and pay T_i to go to v_i" or "start at v_i and pay T_i to go to u_i." We'll label them i_in_u for crossing i forward, i_in_v for crossing i backward. We want to pick exactly one "in" for each i. Then connect them in some order from 1 to N. This might be a typical TSP in a 2K set, but we only want to pick exactly one vertex from each pair. That suggests a bit more complicated approach. 
    #
    #    Alternatively, a standard TSP on K (not 2K) edges, but at each "visit" we can choose forward or backward crossing. Then the distance from e_i (with chosen direction) to e_j (with chosen direction) is either dist(endOf e_i -> startOf e_j) + T(e_j). But we must store for e_i two possibilities for "endOf e_i" depending on direction chosen. This can be integrated into a DP with dimension dp[i][mask][directionOf i], which can get complicated. But we can do a simpler approach: 
    #
    #    We'll do a standard TSP on the set of edges (1..K) with 2^K states. We'll store dp[mask][i][dir], meaning minimal cost to have visited edges in 'mask', with the last used edge = i, used in direction 'dir' (0 or 1), so that we end at the appropriate endpoint. Then to go next to edge j in direction dir', the cost is dist( endOf(e_i, dir), startOf(e_j, dir') ) + T_j. We'll finalize with + dist(endOf(e_k, dir), N). We'll also need to add dist(1, startOf(e_i, dir_i)) + T_i in the beginning. 
    #
    #    This DP would have dimension 2^K * K * 2. That's at most 32 * 5 * 2 = 320 states, transitions up to 5 => ~1600 per query. Then times Q=3000 => 4.8 million => might be feasible if implemented carefully. 
    #
    # 23) We'll do the Floyd-Warshall or something to get dist[u][v]. Then solve each query with the DP approach. Implementation outline:
    #
    # Steps: 
    # A) Build adjacency. 
    # B) Because N=400 is not super large, implement Floyd-Warshall in an efficient manner, or run D' from each node to get dist. Let's do Floyd-Warshall with adjacency initialization might be simpler but must be carefully implemented to pass time. We'll store dist[u][v] in a 2D array, initially large infinity except dist[u][u]=0, and dist[u][v] = min(T_i) if there's a direct edge. If there are multiple edges connecting the same pair, we keep the minimal T_i. Then run Floyd-Warshall. 
    #
    # B') Time complexity is ~N^3=64 million, borderline. We'll try to implement it in Python with potential micro-optimizations. Alternatively, we can do repeated D' for each of the 400 vertices => O(N * (M log N)) = 400*(200000* log 400) ~ 400*(200000 * 8.6) ~ 688 million => bigger. Floyd-Warshall might be safer in well-optimized Python. 
    #
    # C) For each query: 
    #    - let the required edges be e_1..e_K. We'll store (u_i, v_i, t_i). 
    #    - We'll define a DP array dp[mask][last][dir], where 0 <= last < K, dir in {0,1}. 
    #      dp[mask][last][dir] = minimal cost to have used edges in 'mask', with the last edge used being 'last' in direction 'dir'. 
    #    - We'll also define an array to store cost to start an edge i in direction dir from 1: # startCost[i][dir] = dist[1][start(i,dir)] + t_i
    #      and cost to finish from edge i in direction dir to N: # endCost[i][dir] = dist[end(i,dir)][N]
    #    - We'll define an array to store the cost to go from finishing edge i in direction dir to starting edge j in direction dir': # transCost[i][dir][j][dir'] = dist[end(i,dir)][start(j,dir')] + t_j
    #
    #   Then the answer is min_{i,dir} dp[(1<<K)-1][i][dir] + endCost[i][dir], with dp[1<<i][i][dir] = startCost[i][dir].
    #
    #   We'll do the standard TSP bitmask transitions with complexity K * 2^K * K * 2 => 5 * 32 * 5 * 2=1600 steps per query. That times Q=3000 => 4.8 million => hopefully feasible.
    #
    # Implementation details:
    #   - We'll compute dist using Floyd-Warshall in O(N^3). 
    #   - Then for each query:
    #       parse K, the edges. Build arrays start(i,0) = u_i, start(i,1) = v_i, end(i,0) = v_i, end(i,1) = u_i.
    #       Compute startCost[i][dir] = dist[1][start(i,dir)] + t_i
    #       Compute endCost[i][dir] = dist[end(i,dir)][N]
    #       Compute transCost[i][dir][j][dir'] = dist[end(i,dir)][start(j,dir')] + t_j
    #       Then run the DP. 
    #
    #   We'll store dp in a 2D array of size 2^K x K*2, or a dictionary if K < 5. Then we do an outer loop over mask, an inner loop over last,dir in mask, an innermost loop over next not in mask. 
    # 
    #   Implementation detail for Floyd-Warshall:
    #       We'll store dist in a 2D list of length N+1. We'll 1-index. Initialize dist[i][i]=0. For each edge (u,v,t), dist[u][v] = dist[v][u] = min(dist[u][v], t).
    #       Then do for k in range(1..N): 
    #           for i in range(1..N):
    #               dki = dist[i][k]
    #               for j in range(1..N):
    #                   if dki + dist[k][j] < dist[i][j]:
    #                       dist[i][j] = dki + dist[k][j]
    #
    # We'll implement carefully and hope it runs in time. 
    #
    # Let's implement now.

    INF = 10**18

    # Initialize dist
    dist = [[INF]*(N+1) for _ in range(N+1)]
    # Dist[i][i] = 0
    for i in range(1, N+1):
        dist[i][i] = 0

    # Populate from edges
    # If multiple edges connect the same (u,v), keep the minimal T
    for (u, v, t) in bridges:
        if t < dist[u][v]:
            dist[u][v] = t
            dist[v][u] = t

    # Floyd-Warshall
    for k in range(1, N+1):
        dist_k = dist[k]
        for i in range(1, N+1):
            dist_i_k = dist[i][k]
            # Optimization: if dist_i_k == INF, skip
            if dist_i_k == INF:
                continue
            dist_i = dist[i]
            for j in range(1, N+1):
                alt = dist_i_k + dist_k[j]
                if alt < dist_i[j]:
                    dist_i[j] = alt

    # Now handle queries
    # We'll parse them from the input data in the order they come.
    out = []
    from math import inf

    for _ in range(Q):
        K_i = int(input_data[idx]); idx += 1
        reqEdges = [int(x) for x in input_data[idx:idx+K_i]]
        idx += K_i

        # Edges are 1-based index in the input, so subtract 1 to get 0-based
        # We'll gather (u[i], v[i], t[i]) from bridges
        # We'll define for each i: start(i,0) = u_i, end(i,0)=v_i, start(i,1)=v_i, end(i,1)=u_i
        # We'll build arrays of length K_i
        u_list = []
        v_list = []
        t_list = []
        for e_id in reqEdges:
            (u0, v0, t0) = bridges[e_id-1]
            u_list.append(u0)
            v_list.append(v0)
            t_list.append(t0)

        # If K_i == 1, we can handle quickly, but let's just do the general DP approach.

        # Precompute startCost[i][dir], endCost[i][dir], transCost[i][dir][j][dir']
        startCost = [[INF]*2 for _ in range(K_i)]
        endCost = [[INF]*2 for _ in range(K_i)]
        transCost = [[ [INF]*2 for _ in range(K_i)] for __ in range(2)]  # transCost[dir_i][i][dir_j][j]

        for i in range(K_i):
            ui, vi, ti = u_list[i], v_list[i], t_list[i]
            # direction 0: cross ui->vi
            startCost[i][0] = dist[1][ui] + ti if dist[1][ui] < INF else INF
            endCost[i][0] = dist[vi][N] if dist[vi][N] < INF else INF
            # direction 1: cross vi->ui
            startCost[i][1] = dist[1][vi] + ti if dist[1][vi] < INF else INF
            endCost[i][1] = dist[ui][N] if dist[ui][N] < INF else INF

        # Now transCost
        # transCost[i][dirI][j][dirJ] = dist[endOf e_i with dirI][startOf e_j with dirJ] + t_j
        # endOf e_i with dirI = (vi if dirI=0 else ui)
        # startOf e_j with dirJ = (u_j if dirJ=0 else v_j)
        for i in range(K_i):
            ui, vi, ti = u_list[i], v_list[i], t_list[i]
            end_i_0 = vi
            end_i_1 = ui
            for j in range(K_i):
                uj, vj, tj = u_list[j], v_list[j], t_list[j]
                start_j_0 = uj
                start_j_1 = vj
                # i,0 -> j,0
                d_val = dist[end_i_0][start_j_0]
                transCost[0][i][0][j] = d_val + tj if d_val < INF else INF
                # i,0 -> j,1
                d_val = dist[end_i_0][start_j_1]
                transCost[0][i][1][j] = d_val + tj if d_val < INF else INF
                # i,1 -> j,0
                d_val = dist[end_i_1][start_j_0]
                transCost[1][i][0][j] = d_val + tj if d_val < INF else INF
                # i,1 -> j,1
                d_val = dist[end_i_1][start_j_1]
                transCost[1][i][1][j] = d_val + tj if d_val < INF else INF

        # DP: dp[mask][i][dir]
        # mask: bitmask of which edges have been used
        # i: index of last edge used in [0..K_i-1]
        # dir: direction used for that last edge
        # We'll store INF if unreachable
        dp = [[[INF]*2 for _ in range(K_i)] for __ in range(1<<K_i)]

        # Initialize: using only edge i in direction dir
        # dp[1<<i][i][dir] = startCost[i][dir]
        for i in range(K_i):
            for dir in range(2):
                dp[1<<i][i][dir] = startCost[i][dir]

        # BFS the states
        for mask in range(1<<K_i):
            # For each i in [0..K_i-1] that is in mask
            # for each dir in [0..1]
            # if dp[mask][i][dir] < INF, try to add a new edge j not in mask
            mpop = dp[mask]
            for i in range(K_i):
                for dirI in range(2):
                    base_cost = mpop[i][dirI]
                    if base_cost == INF:
                        continue
                    # try any j not in mask
                    not_used = ((1<<K_i)-1) ^ mask  # the complement in [0..K_i-1]
                    # We'll iterate over j in that set
                    sub = not_used
                    while sub:
                        j = (sub & -sub).bit_length()-1  # get the lowest set bit
                        sub ^= (1<<j)
                        # j's direction can be 0 or 1
                        cost_i_j_0 = transCost[dirI][i][0][j]
                        cost_i_j_1 = transCost[dirI][i][1][j]
                        dp_new_mask = mask | (1<<j)
                        dp_cur = dp[dp_new_mask][j]
                        c0 = base_cost + cost_i_j_0
                        c1 = base_cost + cost_i_j_1
                        if c0 < dp_cur[0]:
                            dp_cur[0] = c0
                        if c1 < dp_cur[1]:
                            dp_cur[1] = c1
                    # If K_i < 5, an alternative is a naive loop over j. This bit trick is standard but we can do:
                    # for j in range(K_i):
                    #   if not (mask & (1<<j)):
                    #       cost_i_j_0 = transCost[dirI][i][0][j]
                    #       cost_i_j_1 = transCost[dirI][i][1][j]
                    #       ...

        # Finally, we want min over dp[(1<<K_i)-1][i][dir] + endCost[i][dir].
        full_mask = (1<<K_i)-1
        ans = INF
        for i in range(K_i):
            for dir in range(2):
                cost_val = dp[full_mask][i][dir]
                if cost_val == INF:
                    continue
                cost_val += endCost[i][dir]
                if cost_val < ans:
                    ans = cost_val

        if ans >= INF:
            ans = -1  # theoretically shouldn't happen as the graph is connected
        out.append(str(ans))

    print("
".join(out))