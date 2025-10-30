import sys

# Constants for segment tree initialization
# N_MAX is 4 * 10^5, M_MAX is 2 * 10^5
# Default R value (infinity) should be > N_MAX
# Default person_idx (infinity) should be > M_MAX
DEFAULT_R_INF = 400001
DEFAULT_PERSON_IDX_INF = 200000

class SegmentTree:
    def __init__(self, size, default_r_val, default_person_idx):
        self.size = size
        # Tree array needs to cover up to `size` elements (e.g., towns 1..N or person indices 0..M-1).
        # We use 1-based indexing for towns and 0-based for people indices for internal segment tree.
        # So segment tree will effectively operate on range [1, N] or [0, M-1].
        # For N=4e5, 4*N is 1.6M nodes. For M=2e5, 4*M is 0.8M nodes.
        self.tree = [(default_r_val, default_person_idx)] * (4 * size)
        self.default_val = (default_r_val, default_person_idx)

    def _combine(self, val1, val2):
        # When comparing (R, person_idx), prioritize smaller R.
        # If R's are equal, prioritize smaller person_idx.
        if val1[0] < val2[0]:
            return val1
        elif val2[0] < val1[0]:
            return val2
        else: # R values are equal
            return (val1[0], min(val1[1], val2[1]))

    def update(self, node, start, end, idx, val):
        # idx is the L-value of the town or person index to update
        if start == end:
            # For update, we want to store the minimum value at this leaf
            self.tree[node] = self._combine(self.tree[node], val)
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                self.update(2 * node, start, mid, idx, val)
            else:
                self.update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = self._combine(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, node, start, end, l, r):
        # Query for range [l, r]
        if r < start or end < l: # Query range is outside current node's range
            return self.default_val
        if l <= start and end <= r: # Current node's range is completely within query range
            return self.tree[node]
        
        mid = (start + end) // 2
        p1 = self.query(2 * node, start, mid, l, r)
        p2 = self.query(2 * node + 1, mid + 1, end, l, r)
        return self._combine(p1, p2)

def solve():
    N, M, Q = map(int, sys.stdin.readline().split())

    people_data = [] # Stores (L, R, original_person_idx)
    for i in range(M):
        S, T = map(int, sys.stdin.readline().split())
        L = min(S, T)
        R = max(S, T)
        people_data.append((L, R, i)) # i is the 0-indexed person ID

    queries = [] # Stores (L_k, R_k) as 1-indexed person ranges
    for _ in range(Q):
        queries.append(list(map(int, sys.stdin.readline().split())))

    # 1. Build next_crossing array
    # next_crossing[i] stores the smallest j > i such that person i and j form a crossing pair.
    # If no such j exists, it stores DEFAULT_PERSON_IDX_INF.
    next_crossing = [DEFAULT_PERSON_IDX_INF] * M

    # Segment tree to find crossing intervals. It operates on town indices (1 to N).
    # Stores (min_R_val, person_idx) for intervals starting at L_val.
    st_crossing = SegmentTree(N, DEFAULT_R_INF, DEFAULT_PERSON_IDX_INF)

    for i in range(M - 1, -1, -1): # Iterate backwards from M-1 down to 0
        current_L, current_R, original_person_idx = people_data[i]
        
        # Query range for potential crossing partners: (current_L+1, current_R-1)
        # This range refers to L-values of other intervals.
        query_range_start = current_L + 1
        query_range_end = current_R - 1

        # Check for crossing intervals only if the intermediate range is valid
        # |S_i - T_i| > 1 guarantees current_R - current_L > 1, so query_range_start <= query_range_end
        
        min_R_found, min_person_idx_found = st_crossing.query(1, 1, N, query_range_start, query_range_end)
        
        if min_R_found < current_R:
            # A crossing pair (original_person_idx, min_person_idx_found) is found.
            # current_L < min_person_idx_found.L (implied by query range)
            # min_person_idx_found.R < current_R (condition)
            # This is the L_i < L_j < R_j < R_i crossing pattern.
            next_crossing[original_person_idx] = min_person_idx_found
        
        # Update segment tree with current person's data at their L-value.
        # This makes current person available as a potential 'j' for future 'i's.
        st_crossing.update(1, 1, N, current_L, (current_R, original_person_idx))

    # 2. Build Segment Tree for Range Minimum Query on next_crossing array
    # This segment tree operates on person indices (0 to M-1).
    # It stores the smallest person_idx (from next_crossing values) in a given range.
    st_rmq = SegmentTree(M, DEFAULT_PERSON_IDX_INF, DEFAULT_PERSON_IDX_INF) 
    for i in range(M):
        # For RMQ, the 'R_val' part of the tuple is next_crossing[i], and 'person_idx' is also next_crossing[i].
        # The _combine method will effectively just take the minimum of next_crossing values.
        st_rmq.update(1, 0, M - 1, i, (next_crossing[i], next_crossing[i]))

    # 3. Process Queries
    results = []
    for q_L, q_R in queries:
        # Convert 1-indexed query range (for people) to 0-indexed
        query_start_idx = q_L - 1
        query_end_idx = q_R - 1

        # Query for minimum next_crossing value in the range [query_start_idx, query_end_idx]
        min_val_in_range, _ = st_rmq.query(1, 0, M - 1, query_start_idx, query_end_idx)

        # If min_val_in_range <= query_end_idx, it means there's a person 'i' in the query range
        # whose next_crossing[i] points to a 'j' that is also within the query range [query_start_idx, query_end_idx].
        # This signifies a crossing pair within the queried set of people, making it impossible.
        if min_val_in_range <= query_end_idx:
            results.append("No")
        else:
            results.append("Yes")
    
    sys.stdout.write("
".join(results) + "
")

# Call the solve function
if __name__ == '__main__':
    solve()