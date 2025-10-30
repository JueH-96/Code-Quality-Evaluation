# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N, M, X1 = map(int, sys.stdin.readline().split())
    M = int(M)
    X1 = int(X1)
    trains = []
    for _ in range(M):
        A_i, B_i, S_i, T_i = map(int, sys.stdin.readline().split())
        trains.append({'A': A_i, 'B': B_i, 'S': S_i, 'T': T_i})

    # Variables: x_i (for i from 0 to M-1)
    # Constraints: x_j >= x_i + c_{ij}
    # Edges: from i to j with weight c_{ij}
    # Build graph edges
    edges = []
    lower_bounds = [0] * M
    lower_bounds[0] = X1  # x_0 >= X1

    # Collect departures and arrivals for each city
    from collections import defaultdict

    departures = defaultdict(list)  # departures[c]: list of (S_j, j)
    arrivals = defaultdict(list)    # arrivals[c]: list of (T_i, i)

    for idx, train in enumerate(trains):
        A_i = train['A']
        B_i = train['B']
        S_i = train['S']
        T_i = train['T']
        departures[A_i].append((S_i, idx))
        arrivals[B_i].append((T_i, idx))

    # Build edges
    total_edges = 0
    for c in range(1, N + 1):
        if c not in departures and c not in arrivals:
            continue
        dep_list = departures.get(c, [])
        arr_list = arrivals.get(c, [])

        # Sort departures and arrivals
        dep_list.sort()
        arr_list.sort()

        S_list = [s for s, idx in dep_list]
        dep_indices = [idx for s, idx in dep_list]
        T_list = [t for t, idx in arr_list]
        arr_indices = [idx for t, idx in arr_list]

        # For each arrival, find departures with S_j >= T_i
        for T_i, i_idx in arr_list:
            idx = bisect.bisect_left(S_list, T_i)
            # Process all departures with S_j >= T_i
            for j in range(idx, len(dep_list)):
                S_j, j_idx = dep_list[j]
                # Constraint: x_j >= x_i + c_{ij}
                c_ij = - (S_j - T_i)
                edges.append((i_idx, j_idx, c_ij))
                total_edges += 1

    # Now we can perform Bellman-Ford algorithm
    # Variables x_i, constraints x_j >= x_i + c_{ij}
    # Edges from i to j with weight c_{ij} = - (S_j - T_i)

    M = len(trains)
    x = [float('inf')] * M
    x[0] = lower_bounds[0]
    for i in range(1, M):
        x[i] = lower_bounds[i]

    updated = True
    for _ in range(M):
        updated = False
        for u, v, w in edges:
            new_x_v = max(x[u] + w, lower_bounds[v])
            if x[v] > new_x_v:
                x[v] = new_x_v
                updated = True
    # Check for negative cycles (not needed since problem says solution exists and unique)
    # Now output x_i for i from 1 to M-1 (excluding x_0)
    ans = []
    for i in range(1, M):
        ans.append(int(x[i]))
    print(' '.join(map(str, ans)))

threading.Thread(target=main).start()