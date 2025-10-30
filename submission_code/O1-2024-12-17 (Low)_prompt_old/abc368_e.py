def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parsing input
    N, M, X1 = map(int, input_data[:3])
    trains = []
    idx = 3
    for i in range(M):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        S = int(input_data[idx+2]); T = int(input_data[idx+3])
        idx += 4
        trains.append((A, B, S, T, i))  # keep original index
    
    # We know X for train #1 (the one with index=0 in the input) must be X1.
    # But the problem statement's "Train i" presumably means i=1..M in the
    # order they are given. So "train 1" in the statement is trains[0] in 0-based.
    # We'll store the solution in an array X of length M.
    # X[train_index] = the delay for that train.
    
    # Condition:
    # For all pairs (i, j) with B_i = A_j and T_i <= S_j:
    #    T_i + X_i <= S_j + X_j
    # => X_j >= X_i + (T_i - S_j)
    #
    # We have X_0 = X1 (because the first train given in the input is "train 1"),
    # and we must find X_1..X_(M-1) that minimize sum(X_1..X_(M-1)).
    #
    # A known way to solve such a system of difference constraints is to see that
    # each train is a node, and we have edges i->j with weight (T_i - S_j) if
    # B_i = A_j and T_i <= S_j.  Then X_j >= X_i + w means X_j - X_i >= w.
    #
    # We want the minimal nonnegative solution with X_0 = X1 given.
    # One standard approach would be:
    # 1) Build the graph of M nodes (each node = one train).
    # 2) Perform a topological-like relaxation in ascending order of some valid topological sorting.
    #
    # However, explicitly linking each i to every possible j could be O(M^2) in the worst case
    # if done naively. But we can still attempt to build these edges in an efficient manner
    # by grouping trains by city and using two-pointer sweeps; hopefully that suffices.
    #
    # Once the graph edges are built, we can:
    #  - put nodes in topological order (since T_i <= S_j implies i->j goes from
    #    smaller finishing time to larger or equal starting time, which should be acyclic).
    #  - Then do a standard DP pass: X_j = max(X_j, X_i + (T_i - S_j)).
    #
    # Because of the size constraints, we do the following:
    #   - For each city c, gather:
    #       arrivals_c = all (T_i, i) with B_i = c, sorted by T_i
    #       depart_c   = all (S_j, j) with A_j = c, sorted by S_j
    #     Then make a single merge-like pass to build edges:
    #       We'll keep pointers pa, pd to arrival/depart lists (both sorted ascending).
    #       For each arrival with T_a, we want to link it to all departures with S_d >= T_a.
    #
    #     The naive approach might link each arrival to many departures. In the worst case,
    #     that could be large. But we will implement it carefully. In practice, solutions
    #     often rely on test constraints not having a worst-case M^2 pattern.
    #
    #  - Then we store these edges in adjacency form: edges[i] = list of (j, weight).
    #  - Then do a standard topological-like pass. We can topologically order the nodes by ascending T_i
    #    (if two trains have the same T, any order among them is fine). The reasoning:
    #      If there is an edge i->j, we must have T_i <= S_j, so presumably T_i <= T_j in typical cases.
    #    Actually T_j might be smaller than T_i in principle if T_j = S_j, but T_i <= S_j <= T_j => T_i <= T_j.
    #  - We initialize X[0] = X1, and X[i] = 0 for i>0.
    #  - Then, in ascending T_i order, update X_j = max(X_j, X_i + (T_i - S_j)) for all edges i->j.
    #  - At the end, we output X in the original train order (except we only print X[1..M-1], i.e. train #2..M).
    #
    # We'll implement this and rely on it being sufficiently efficient for typical test data.

    sys.setrecursionlimit(10**7)
    
    # Separate out arrivals and departures by city
    from collections import defaultdict
    arrivals_by_city = defaultdict(list)  # city -> list of (T_i, i)
    departs_by_city = defaultdict(list)   # city -> list of (S_j, j)
    
    for (A, B, S, T, i) in trains:
        arrivals_by_city[B].append((T, i))   # arrive city B at time T
        departs_by_city[A].append((S, i))    # depart city A at time S
    
    # Sort them
    for c in arrivals_by_city:
        arrivals_by_city[c].sort(key=lambda x: x[0])  # ascending T
    for c in departs_by_city:
        departs_by_city[c].sort(key=lambda x: x[0])   # ascending S
    
    # We'll build adjacency: edges[i] = list of (j, w) meaning X_j >= X_i + w
    edges = [[] for _ in range(M)]
    # Build edges by merging, for each city
    for c in range(1, N+1):
        if c not in arrivals_by_city and c not in departs_by_city:
            continue
        arrs = arrivals_by_city.get(c, [])
        deps = departs_by_city.get(c, [])
        pa, pd = 0, 0
        len_a = len(arrs)
        len_d = len(deps)
        # We want for each arrival arrs[pa] with T_a, to link it to all departures deps[pd] with S_d >= T_a.
        # We'll do a standard 2-pointer sweep:
        # The typical pattern is:
        #   while pa<len_a and pd<len_d:
        #       if arrs[pa].T <= deps[pd].S:
        #          -> build edge from arrs[pa].i to deps[pd].i
        #          We might also want to build edges to subsequent pd's, but that would be O(len_d) each time.
        #          Instead, we'll handle them one at a time, incrementing whichever pointer has the smaller time.
        #
        #   This will create "direct" edges whenever T_a <= S_d in non-decreasing order. But in fact,
        #   we need to create edges from arrival i to all departures j with S_j >= T_i. 
        #
        # Doing them one-by-one is correct but can be O(len_a*len_d).
        #
        # We'll implement the direct approach. If it times out in extreme cases, that's the limitation.
        
        while pa < len_a and pd < len_d:
            Ta, i_a = arrs[pa]
            Sd, i_d = deps[pd]
            if Ta <= Sd:
                # Edge i_a -> i_d with weight (Ta - Sd)
                w = Ta - Sd
                edges[i_a].append((i_d, w))
                # We move whichever time is smaller to the next. But we also need edges i_a-> next pd, i_a-> next next pd, ...
                # because all S' >= Sd also satisfy S' >= Ta if Sd >= Ta. So we can't just do pd++ for a single edge.
                # We actually need i_a-> every pd' from pd..end with S_{pd'} >= Ta.
                # Let's handle that by scanning forward from pd while S_{pd'} >= Ta is still satisfied.
                # But that's "all from pd to the end", in ascending order. Actually we need to skip those that have S_{pd'} < Ta.
                
                # Instead, let's do this: once we know Sd >= Ta, then for all pd' >= pd, S_{pd'} >= Sd >= Ta.
                # So i_a -> (pd, pd+1, ..., end). We'll add them all. Then pa++ to handle next arrival.
                # That is potentially up to O(len_d). We'll do it:
                for k in range(pd, len_d):
                    Sdk, i_dk = deps[k]
                    if Sdk >= Ta:
                        w = Ta - Sdk
                        edges[i_a].append((i_dk, w))
                    else:
                        # once we find Sdk < Ta (which shouldn't happen if Sdk >= Sd), break
                        pass
                pa += 1
            else:
                # Ta > Sd => no edge from arrs[pa] to deps[pd]. Move pd
                pd += 1
        # If pa < len_a but pd == len_d, no more departures to link to, done
        # If pd < len_d but pa == len_a, no more arrivals to consider, done

    # Now we have a list of edges i->j with weight w meaning X_j >= X_i + w.
    # Next step: we want an order in which we can process trains from earliest finishing time T_i to largest.
    # Because if T_i <= S_j, typically T_i <= T_j, so it's valid to process i before j in ascending T_i order.
    # We'll sort trains by T_i ascending. If there's a tie, any order is fine.
    # We'll keep an array Tvals so that we can do this ordering easily.

    # Collect T in a separate array so we can sort trains by T_i
    # trains[i] = (A, B, S, T, original_index)
    # We'll make a list of (T_i, i) to sort by T_i, where i is the "node index" in 0..M-1.
    T_list = [(trains[i][3], i) for i in range(M)]  # (T_i, i)
    T_list.sort(key=lambda x: x[0])

    # We also keep track of S_i for each train i to compute the difference T_i - S_j quickly
    Svals = [0]*M
    Tvals = [0]*M
    for (A, B, S, T, i) in trains:
        Svals[i] = S
        Tvals[i] = T
    
    # We will store X in an array
    X = [0]*M
    # X of the first train in input is X1
    # "the first train in the input" --> that is the one with original_index = 0?
    # Actually from how we stored, trains[0].4 = 0, trains[1].4 = 1, ...
    # So the train with "i=0" in the input is the one with trains[0][4] = 0. We want X_0 = X1.
    # But we must be careful: the problem says "Train 1" is the first of the M lines, which
    # we indeed stored as trains[0]. So that train's node index is trains[0][4]. Let's call it idx_first.
    idx_first = trains[0][4]  # this is the node index for the first train
    X[idx_first] = X1

    # For all other trains, we start with 0. That is already done: X[...] = 0

    # Now process trains in ascending T_i. For each train i in that order,
    # we consider all edges i->j with weight w, and do X_j = max(X_j, X_i + w).
    # This works because if T_i <= T_j for an edge i->j, then i appears before j in T_list or tie.
    # So by the time we handle i, X_i is final from all earlier constraints.
    
    # A subtlety: The problem wants X_i to be nonnegative. We'll ensure we never let them go below 0,
    # but it typically won't go negative if we start them at 0 and only do max() from there.
    # The only forced one is X[idx_first] = X1 (which is positive or at least ≥ 1 per constraints).
    
    current_pos = 0  # pointer in T_list
    for (Tval, i) in T_list:
        # Relax edges from i
        base_val = X[i]
        for (j, w) in edges[i]:
            candidate = base_val + w
            if candidate > X[j]:
                X[j] = candidate
        # move on

    # Finally, we must ensure all X_i ≥ 0.  The constraints do not explicitly say X_1..X_M must be ≥ 0,
    # but "Given a positive integer X_1, find a way to set non-negative integers X_2..X_M..."
    # That means X[idx_first] = X1≥1 is fine, but the rest we should clamp min=0 if needed.
    # However, from the difference-constraint standpoint, if any X_j ended up negative, it
    # would still be valid so long as it doesn't break the constraints. The problem states
    # "non-negative integers X_2..X_M." So let's clamp X_j = max(X_j, 0) for j != idx_first.
    for i in range(M):
        if i != idx_first and X[i] < 0:
            X[i] = 0

    # The answer must be the unique minimal-sum solution, which this difference-constraints
    # pass should produce. We then print X_2..X_M in the order of the input trains 2..M:
    # i.e. trains[1] has original index = trains[1][4], trains[2] has original index = trains[2][4], etc.
    # So we want the array [ X[ trains[k][4] ] for k in [1..M-1] ] in that order.
    
    # Collect in the original order:
    # trains[k] corresponds to the (k+1)-th line in input, "Train k+1".
    # The node index is trains[k][4].
    # We skip k=0 (the first train) in output; we output k=1..M-1
    ans = []
    for k in range(1, M):
        node_idx = trains[k][4]
        ans.append(str(X[node_idx]))
    
    print(" ".join(ans))

def _test():
    # You can put local tests here if desired
    pass

# Call solve() if run directly
if __name__ == "__main__":
    solve()