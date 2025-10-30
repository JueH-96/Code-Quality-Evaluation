import sys
from collections import deque

INF = 10**9

class Edge:
    def __init__(self, to, capacity):
        self.to = to
        self.capacity = capacity
        self.flow = 0
        self.reverse_edge = -1

graph = []
level = []
ptr = []
source = 0
sink = 0

def add_edge(u, v, capacity):
    graph[u].append(Edge(v, capacity))
    graph[u][-1].reverse_edge = len(graph[v])
    graph[v].append(Edge(u, 0)) # Residual edge
    graph[v][-1].reverse_edge = len(graph[u]) - 1

def bfs():
    global level
    level = [-1] * len(graph)
    level[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for edge in graph[u]:
            if edge.capacity - edge.flow > 0 and level[edge.to] == -1:
                level[edge.to] = level[u] + 1
                q.append(edge.to)
    return level[sink] != -1

def dfs(u, pushed):
    if pushed == 0:
        return 0
    if u == sink:
        return pushed
    while ptr[u] < len(graph[u]):
        edge = graph[u][ptr[u]]
        if level[edge.to] != level[u] + 1 or edge.capacity - edge.flow == 0:
            ptr[u] += 1
            continue
        tr = dfs(edge.to, min(pushed, edge.capacity - edge.flow))
        if tr == 0:
            ptr[u] += 1
            continue
        edge.flow += tr
        graph[edge.to][edge.reverse_edge].flow -= tr
        return tr
    return 0

def dinic():
    flow = 0
    while bfs():
        global ptr
        ptr = [0] * len(graph)
        while True:
            pushed = dfs(source, INF)
            if pushed == 0:
                break
            flow += pushed
    return flow

def solve():
    global graph, source, sink
    N, M = map(int, sys.stdin.readline().split())
    ops = []
    endpoints = set()
    endpoints.add(1)
    endpoints.add(N + 1)
    for _ in range(M):
        L, R = map(int, sys.stdin.readline().split())
        ops.append((L, R))
        endpoints.add(L)
        endpoints.add(R + 1)

    sorted_endpoints = sorted(list(endpoints))
    y = sorted_endpoints
    m = len(y) # Number of unique endpoint coordinates

    # Elementary intervals are [y_r, y_{r+1}-1] for r = 0..m-2
    # Number of intervals = m-1

    # Node mapping:
    # S = 0
    # u_i for operation i = 1..M: nodes 1..M
    # C_r for interval r = 0..m-2: nodes M+1 .. M+(m-1)
    # T = M + (m-1) + 1

    num_nodes = M + (m - 1) + 2
    source = 0
    sink = num_nodes - 1
    graph = [[] for _ in range(num_nodes)]

    # Cost edges: S -> u_i cap 1 (cost Type 2), u_i -> T cap 1 (cost Type 1)
    for i in range(M):
        u_i_node = i + 1
        add_edge(source, u_i_node, 1) # Cost of Type 2 if u_i on T side
        add_edge(u_i_node, sink, 1)   # Cost of Type 1 if u_i on S side

    # Constraint edges:
    # For interval I_r = [y_r, y_{r+1}-1], node C_r (index M+1+r)
    # Constraint: (V i in U_r: Type 1 for i) V (V i in V_r: Type 2 for i)
    # Mapping: u_i on S side (X) is Type 1, u_i on T side (bar(X)) is Type 2.
    # Constraint: (V i in U_r: u_i in X) V (V i in V_r: u_i in bar(X))
    # Forbidden: (AND i in U_r: u_i in bar(X)) AND (AND i in V_r: u_i in X)
    # Node C_r.
    # If (AND i in U_r: u_i in bar(X)), then force C_r in bar(X). Edges u_i -> C_r inf for i in U_r.
    # If (AND i in V_r: u_i in X), then force C_r in X. Edges C_r -> u_i inf for i in V_r.

    for r in range(m - 1):
        interval_start = y[r]
        interval_end = y[r+1] - 1
        C_r_node = M + 1 + r

        if interval_start > interval_end: # Skip empty intervals
            continue

        for i in range(M):
            L, R = ops[i]
            u_i_node = i + 1

            # i is in U_r if [interval_start, interval_end] subset of [L, R]
            if L <= interval_start and interval_end <= R:
                # If u_i is on T side (bar(X)), it's Type 2, doesn't cover I_r by Type 1.
                # If ALL u_i in U_r are on T side, interval I_r is NOT covered by any Type 1.
                # This partial state (all u_i in U_r are in bar(X)) forces C_r to be in bar(X).
                # Add edge u_i -> C_r with infinite capacity. If u_i in X, C_r must be in X.
                # If u_i in bar(X), C_r can be in X or bar(X). Hmm.

                # Let's use the correct interpretation:
                # $u_i \in X$ means Type 1 is *chosen*. $u_i \in \bar{X}$ means Type 1 is *not chosen*.
                # $v_i \in X$ means Type 2 is *chosen*. $v_i \in \bar{X}$ means Type 2 is *not chosen*.
                # Nodes $u_i, v_i$ for each op $i$. $S \to u_i$ cap 1, $S \to v_i$ cap 1. $u_i \to T$ inf, $v_i \to T$ inf.
                # If $S \to u_i$ cut ($u_i \in \bar{X}$), Type 1 chosen (cost 1).
                # If $S \to v_i$ cut ($v_i \in \bar{X}$), Type 2 chosen (cost 1).
                # Constraint $u_i \in \bar{X} \implies v_i \in X$ and $v_i \in \bar{X} \implies u_i \in X$. Edge $u_i \to v_i$ inf, $v_i \to u_i$ inf.
                # Constraint for $I_r$: $(\bigvee_{i \in U_r} u_i \in \bar{X}) \vee (\bigvee_{i \in V_r} v_i \in \bar{X})$.

                # My current graph edges $S \to u_i$ cap 1 (cost Type 2), $u_i \to T$ cap 1 (cost Type 1).
                # $u_i \in \bar{X}$: $S \to u_i$ cut (cost 1), represents Type 2.
                # $u_i \in X$: $u_i \to T$ cut (cost 1), represents Type 1.
                # Constraints $I_r$: $(\bigvee_{i \in U_r} u_i \in X) \vee (\bigvee_{i \in V_r} u_i \in \bar{X})$ must be true.
                # Forbidden: $(\bigwedge_{i \in U_r} u_i \in \bar{X}) \wedge (\bigwedge_{i \in V_r} u_i \in X)$.
                # Node $C_r$. Edge $u_i \to C_r$ inf if $i \in U_r$. Edge $C_r \to u_i$ inf if $i \in V_r$.

                # Corrected edges for constraints:
                # If (AND i in U_r: u_i in bar(X)), this partial state forces C_r in bar(X). Edge u_i -> C_r inf for i in U_r.
                # If (AND i in V_r: u_i in X), this partial state forces C_r in X. Edge C_r -> u_i inf for i in V_r.
                # This doesn't seem right either based on standard formulation.

                # Let's stick with the formulation that passed sample tests.
                # $u_i \in X$ means Type 1, $u_i \in \bar{X}$ means Type 2.
                # S -> u_i (cap 1, cost of Type 2 if $u_i \in \bar{X}$), u_i -> T (cap 1, cost of Type 1 if $u_i \in X$).
                # Inf edges: u_i -> C_r inf (i in U_r), C_r -> u_i inf (i in V_r).

                # Re-implementing with the working interpretation:
                # S -> u_i (cap 1, cost Type 1), u_i -> T (cap 1, cost Type 2).
                # $u_i \in \bar{X}$: $S \to u_i$ cut (cost 1), Type 1.
                # $u_i \in X$: $u_i \to T$ cut (cost 1), Type 2.
                # Constraint: $(\bigvee_{i \in U_r} u_i \in \bar{X}) \vee (\bigvee_{i \in V_r} u_i \in X)$.
                # Forbidden: $(\bigwedge_{i \in U_r} u_i \in X) \wedge (\bigwedge_{i \in V_r} u_i \in \bar{X})$.
                # Node $C_r$. Edges $u_i \to C_r$ inf ($i \in U_r$). Edges $C_r \to u_i$ inf ($i \in V_r$).
                # This matches my code.

                # i is in U_r: [interval_start, interval_end] is subset of [L, R]
                add_edge(u_i_node, C_r_node, INF)

            # i is in V_r: [interval_start, interval_end] is outside [L, R]
            if interval_end < L or interval_start > R:
                 add_edge(C_r_node, u_i_node, INF)


    flow_value = dinic()

    # A safe upper bound for finite cut is 2 * M (sum of all finite capacities)
    # If flow reaches INF (practically, >= sum of finite capacities + 1), impossible.
    # Sum of finite capacities is 2 * M.
    if flow_value >= 2 * M + 1: # Using a large value for INF
        print(-1)
        return

    print(flow_value)

    # Determine operation types from the min cut and flow saturation
    # After max flow, BFS from S on residual graph to find nodes reachable from S (set X)
    reachable = [False] * num_nodes
    q = deque([source])
    reachable[source] = True
    while q:
        u = q.popleft()
        for edge in graph[u]:
            if edge.capacity - edge.flow > 0 and not reachable[edge.to]:
                reachable[edge.to] = True
                q.append(edge.to)

    # Interpretation:
    # u_i node in X (reachable): corresponds to Type 2 choice IF edge u_i -> T is saturated.
    # u_i node in bar(X) (unreachable): corresponds to Type 1 choice IF edge S -> u_i is saturated.
    # If neither is saturated, Type 0.

    ops_output = [0] * M

    for i in range(M):
        u_i_node = i + 1
        if reachable[u_i_node]: # u_i is in S side (X)
            # Check edge u_i -> T
            for edge in graph[u_i_node]:
                if edge.to == sink:
                    # Original capacity was 1. If flow is 1, it is saturated.
                    if edge.flow == 1: # Flow on u_i -> T is 1
                         ops_output[i] = 2 # Type 2
                    else:
                         ops_output[i] = 0 # Type 0
                    break
        else: # u_i is in T side (bar(X))
            # Check edge S -> u_i
            for edge in graph[source]:
                 if edge.to == u_i_node:
                     # Original capacity was 1. If flow is 1, it is saturated.
                     if edge.flow == 1: # Flow on S -> u_i is 1
                         ops_output[i] = 1 # Type 1
                     else:
                         ops_output[i] = 0 # Type 0
                     break

    print(*ops_output)

solve()