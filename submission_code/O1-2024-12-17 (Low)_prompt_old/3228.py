class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        """
        We want to keep exactly n/2 elements from nums1 and n/2 from nums2,
        so that the union (as a set) of the remaining elements is as large as possible.
        
        Observing that each distinct element can be contributed by either nums1 or nums2 (if it exists there),
        we can model this as a maximum flow (or bipartite matching with capacity) problem:

        1) Let distinct_vals be all distinct elements from both arrays.
        2) Create a flow network:
           - Source node S connects to each distinct element node (call it x) with capacity 1.
           - Each distinct element x can connect to "node A" (representing array1) if it appears in nums1.
           - Each distinct element x can connect to "node B" (representing array2) if it appears in nums2.
           - Node A and Node B each connect to the sink T with capacity n/2 (because we can only pick n/2 from each array).
           - Every edge from x to A or B has capacity 1. The edges from S to x also have capacity 1.
        3) The maximum flow in this network equals the maximum number of distinct elements we can "choose".
           Each chosen distinct element uses capacity from either array1 or array2 exactly once.

        We'll implement Dinic's algorithm for efficiency. The answer is simply the max flow.
        """

        from collections import defaultdict, deque

        # Step 1: Count frequencies to know which element appears in which array
        freq1 = defaultdict(int)
        freq2 = defaultdict(int)
        for v in nums1:
            freq1[v] += 1
        for v in nums2:
            freq2[v] += 1
        
        distinct_vals = list(set(freq1.keys()).union(freq2.keys()))
        d = len(distinct_vals)  # number of distinct elements

        # Assign an index to each distinct element
        # We'll build the flow network with these node indices:
        #   S = 0
        #   distinct elements: 1..d
        #   A = d+1
        #   B = d+2
        #   T = d+3
        val_to_idx = {}
        for i, val in enumerate(distinct_vals, 1):
            val_to_idx[val] = i
        
        S = 0
        A = d + 1
        B = d + 2
        T = d + 3
        NODES = d + 4  # total number of vertices in the flow network

        # Dinic's algorithm requires an adjacency list of edges.
        # We'll store edges as (to, capacity, rev_index), where rev_index is
        # the index of the reverse edge in the adjacency list of 'to'.
        graph = [[] for _ in range(NODES)]

        def add_edge(u, v, cap):
            # forward edge
            graph[u].append([v, cap, len(graph[v])])
            # reverse edge
            graph[v].append([u, 0, len(graph[u]) - 1])

        # Step 2: Build the graph
        # S -> x has capacity 1
        for val, x_idx in val_to_idx.items():
            add_edge(S, x_idx, 1)

        # x -> A if freq1[x_val] > 0
        # x -> B if freq2[x_val] > 0
        for val, x_idx in val_to_idx.items():
            if freq1[val] > 0:
                add_edge(x_idx, A, 1)
            if freq2[val] > 0:
                add_edge(x_idx, B, 1)

        # A -> T with capacity n/2
        # B -> T with capacity n/2
        n = len(nums1)  # same as len(nums2)
        half = n // 2
        add_edge(A, T, half)
        add_edge(B, T, half)

        # Step 3: Implement Dinic's max flow to find the maximum flow from S to T
        def bfs_level_graph():
            level = [-1]*NODES
            level[S] = 0
            queue = deque([S])
            while queue:
                u = queue.popleft()
                for v, cap, _ in graph[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            return level

        def send_flow(u, f, T, level, it):
            if u == T:
                return f
            while it[u] < len(graph[u]):
                v, cap, rev = graph[u][it[u]]
                if cap > 0 and level[v] == level[u] + 1:
                    curr_flow = min(f, cap)
                    temp_flow = send_flow(v, curr_flow, T, level, it)
                    if temp_flow > 0:
                        # adjust capacities
                        graph[u][it[u]][1] -= temp_flow
                        graph[v][rev][1] += temp_flow
                        return temp_flow
                it[u] += 1
            return 0

        max_flow = 0
        while True:
            level = bfs_level_graph()
            if level[T] < 0:  # no more augmenting path
                break
            it = [0]*NODES
            while True:
                flow_sent = send_flow(S, float('inf'), T, level, it)
                if flow_sent <= 0:
                    break
                max_flow += flow_sent

        return max_flow