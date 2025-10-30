def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast parsing of input
    # N, M, X1 on the first line
    # Then M lines of A_i, B_i, S_i, T_i
    
    # Parse first line
    N = int(input_data[0])
    M = int(input_data[1])
    X1_given = int(input_data[2])
    
    # Read train data
    # We'll store zero-based: trains[i] = (A_i, B_i, S_i, T_i)
    # where i=0 corresponds to the "train #1" in the problem.
    trains = []
    ptr = 3
    for i in range(M):
        A_i = int(input_data[ptr]); B_i = int(input_data[ptr+1])
        S_i = int(input_data[ptr+2]); T_i = int(input_data[ptr+3])
        trains.append((A_i, B_i, S_i, T_i))
        ptr += 4

    # Build "arrivals" and "departures" lists for each city
    # arrivals[c] will hold (T_i, train_index) for all trains ending in city c
    # departures[c] will hold (S_i, train_index) for all trains starting in city c
    # We'll use 1-based indexing for cities, so make lists of size N+1
    arrivals = [[] for _ in range(N+1)]
    departures = [[] for _ in range(N+1)]
    
    for i, (A_i, B_i, S_i, T_i) in enumerate(trains):
        arrivals[B_i].append((T_i, i))
        departures[A_i].append((S_i, i))
    
    # Sort arrivals and departures by their time keys
    for c in range(1, N+1):
        arrivals[c].sort(key=lambda x: x[0])    # sort by T_i ascending
        departures[c].sort(key=lambda x: x[0]) # sort by S_i ascending

    # Build adjacency list: adjacency[i] = list of (j, w) meaning X_j >= X_i + w
    # We'll have at most M edges
    adjacency = [[] for _ in range(M)]
    
    # For each city c, we match each departure with the best (largest T_i) arrival
    # that satisfies T_i <= S_j.  That creates one edge from that arrival-train i
    # to the departure-train j, with weight w = (T_i - S_j).
    for c in range(1, N+1):
        arr = arrivals[c]
        dep = departures[c]
        p = 0
        best_idx = -1
        # arr[p] = (T_i, i), dep[...] = (S_j, j)
        for (S_j, j) in dep:
            while p < len(arr) and arr[p][0] <= S_j:
                best_idx = arr[p][1]  # train index
                p += 1
            if best_idx != -1:
                i = best_idx
                w = trains[i][3] - trains[j][2]  # T_i - S_j
                adjacency[i].append((j, w))
    
    # We now perform a topological-like relaxation in order of ascending T_i.
    # Because if there's an edge i->j, we must have T_i <= S_j < T_j => T_j > T_i,
    # so sorting by T_i is a valid topological order (no forward edge goes to a smaller T).
    order = sorted(range(M), key=lambda i: (trains[i][3], i))  # sort by T_i, then by i to break ties
    
    # X array: X[i] is the delay for train i (0-based), train 0 is "train #1" in the problem.
    X = [0]*(M)
    X[0] = X1_given  # the first train's X_1 is given
    
    # Relax in topological order
    for i in order:
        xi = X[i]
        for (j, w) in adjacency[i]:
            new_val = xi + w
            # We cannot go below 0
            if new_val < 0:
                new_val = 0
            if new_val > X[j]:
                X[j] = new_val
    
    # The problem asks us to print X_2, ..., X_M in that order.
    # In our 0-based indexing, that means we skip X[0] and print X[1..M-1].
    # Join them with space
    print(" ".join(map(str, X[1:])))
    
# Don't forget to call main()
if __name__ == "__main__":
    main()