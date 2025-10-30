# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    intervals = []
    events = []

    for idx in range(Q):
        L_i, R_i, C_i = map(int, sys.stdin.readline().split())
        intervals.append((L_i, R_i, C_i, idx))
        events.append((L_i, 1, C_i))
        events.append((R_i + 1, -1, C_i))

    # Step 1: Compute C_j for each position j
    events.sort()
    C_j = [None] * (N + 2)  # positions from 1 to N
    active_costs = []
    import bisect

    pos = 0
    idx = 0
    current_min = None
    event_idx = 0
    total_cost = 0
    discon = False

    # For detecting uncovered positions
    covered = [False] * (N + 2)

    # Process events
    from collections import defaultdict
    active_costs = []
    positions = []
    for event in events:
        positions.append(event[0])
    positions = sorted(set(positions))
    event_by_pos = defaultdict(list)
    for event in events:
        event_by_pos[event[0]].append(event)

    current_active = []
    import heapq
    heap = []
    removed = {}
    position_set = set()
    pos_idx = 0

    result_Cj = [math.inf] * (N + 2)

    for j in range(1, N + 1):
        while positions and positions[0] <= j:
            p = positions.pop(0)
            for event in event_by_pos[p]:
                if event[1] == 1:
                    bisect.insort_left(active_costs, event[2])
                elif event[1] == -1:
                    idx = bisect.bisect_left(active_costs, event[2])
                    if idx < len(active_costs) and active_costs[idx] == event[2]:
                        active_costs.pop(idx)
        if active_costs:
            C_j = active_costs[0]
            result_Cj[j] = C_j
            total_cost += C_j
            covered[j] = True
        else:
            discon = True
            break

    if discon:
        print(-1)
        return

    # Step 3: Check whether intervals are connected

    intervals.sort()
    cur_R = intervals[0][1]
    discon_intervals = False
    for i in range(len(intervals) - 1):
        if intervals[i + 1][0] <= cur_R + 1:
            cur_R = max(cur_R, intervals[i + 1][1])
        else:
            discon_intervals = True
            break

    if discon_intervals:
        print(-1)
        return

    # Build edge list
    edges = []
    for idx, (L_i, R_i, C_i, interval_id) in enumerate(intervals):
        for j in range(L_i, R_i + 1):
            edges.append((C_i, interval_id + N + 1, j))
    # Since there are no edges between hubs directly, we have to rely on nodes to connect hubs
    # So we need to include all possible edges between hubs and nodes

    # Build Union-Find data structure
    parent = [i for i in range(N + Q + 2)]

    def find(u):
        while u != parent[u]:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        parent[u_root] = v_root
        return True

    # Sort edges by cost
    edges.sort()

    total_MST_cost = 0
    for cost, u, v in edges:
        if union(u, v):
            total_MST_cost += cost

    # Check if all nodes are connected
    groups = set()
    for i in range(1, N + Q + 1):
        groups.add(find(i))

    if len(groups) > 1:
        print(-1)
    else:
        print(total_MST_cost)

threading.Thread(target=main).start()