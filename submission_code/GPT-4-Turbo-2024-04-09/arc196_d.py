def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    from collections import defaultdict
    
    # Read all trips
    trips = []
    for _ in range(M):
        S = int(data[idx])
        idx += 1
        T = int(data[idx])
        idx += 1
        if S > T:
            S, T = T, S
        trips.append((S, T))
    
    # Read all queries
    queries = []
    for _ in range(Q):
        L = int(data[idx]) - 1
        idx += 1
        R = int(data[idx]) - 1
        idx += 1
        queries.append((L, R))
    
    # Sort trips by starting point S, then by ending point T
    trips.sort()
    
    # To answer the queries efficiently, we need to preprocess the trips
    # We need to check if any two trips in the range [L, R] are incompatible
    
    # Incompatibility arises if two trips overlap in such a way that they cannot
    # both have the required stamina changes. This happens if:
    # - One trip is entirely contained within another
    # - Two trips intersect but do not align perfectly in direction
    
    # We will use a segment tree or a similar structure to handle range maximum queries
    # to quickly determine the maximum end point T for any starting range of S.
    
    # We will use a Fenwick Tree (Binary Indexed Tree) to handle this efficiently
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (size + 1)
        
        def update(self, index, value):
            while index <= self.size:
                self.tree[index] = max(self.tree[index], value)
                index += index & -index
        
        def query(self, index):
            max_val = 0
            while index > 0:
                max_val = max(max_val, self.tree[index])
                index -= index & -index
            return max_val
    
    # We need to check the maximum T for any starting S in the range [L, R]
    # We use a Fenwick Tree to keep track of the maximum T for any S
    max_T_for_S = FenwickTree(N)
    
    # We will process the queries offline and sort them by R
    queries = [(L, R, i) for i, (L, R) in enumerate(queries)]
    queries.sort(key=lambda x: x[1])  # Sort by R
    
    # We will process trips in sorted order and update the Fenwick Tree
    results = [None] * Q
    current_trip_index = 0
    max_T_seen = [0] * (N + 1)
    
    for L, R, query_index in queries:
        # Process all trips up to R
        while current_trip_index <= R:
            S, T = trips[current_trip_index]
            max_T_for_S.update(S, T)
            max_T_seen[S] = max(max_T_seen[S], T)
            current_trip_index += 1
            if current_trip_index >= M:
                break
        
        # Now answer the query for range [L, R]
        # We need to check if there's any incompatibility in the range [L, R]
        possible = True
        for i in range(L, R + 1):
            S, T = trips[i]
            # Check if there's any trip starting from S' where L <= S' <= S that ends after T
            max_T_in_range = max_T_for_S.query(S)
            if max_T_in_range > T:
                possible = False
                break
        
        results[query_index] = "Yes" if possible else "No"
    
    # Output all results for each query
    for result in results:
        print(result)