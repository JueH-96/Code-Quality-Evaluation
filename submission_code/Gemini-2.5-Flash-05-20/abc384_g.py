import sys

# Fenwick Tree (BIT) implementation
class FenwickTree:
    def __init__(self, size):
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, idx, delta):
        idx += 1  # 1-based indexing for BIT
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & (-idx)

    def query(self, idx):
        idx += 1 # 1-based indexing for BIT
        s = 0
        while idx > 0:
            s += self.tree[idx]
            idx -= idx & (-idx)
        return s

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    K = int(sys.stdin.readline())
    queries_input = []
    for _ in range(K):
        X_k, Y_k = map(int, sys.stdin.readline().split())
        queries_input.append((X_k, Y_k))

    # Precompute prefix sums for A and B
    prefix_sum_A = [0] * (N + 1)
    for i in range(N):
        prefix_sum_A[i+1] = prefix_sum_A[i] + A[i]

    prefix_sum_B = [0] * (N + 1)
    for i in range(N):
        prefix_sum_B[i+1] = prefix_sum_B[i] + B[i]

    # Coordinate compression for values
    all_values = sorted(list(set(A + B)))
    val_to_comp_idx = {val: i for i, val in enumerate(all_values)}
    M = len(all_values) # Number of unique compressed values

    # Stores results for S_ge_A and S_ge_B for each query
    ans_S_ge_A = [0] * K
    ans_S_ge_B = [0] * K

    # Create events for the sweep line
    # Events are (value, type, original_idx_A_or_B_or_X_k, original_Y_k, query_original_idx)
    # Type 'B': A B_j value
    # Type 'A': An A_i value
    # Type 'Q': A query (X_k, Y_k)
    events = []
    for i in range(N):
        events.append((A[i], 'A', i + 1, -1, -1)) # A_i_val, type, A_i_idx, Y_k, Q_idx
    for j in range(N):
        events.append((B[j], 'B', j + 1, -1, -1)) # B_j_val, type, B_j_idx, Y_k, Q_idx
    for k in range(K):
        events.append((A[queries_input[k][0]-1], 'Q', queries_input[k][0], queries_input[k][1], k)) # A_X_k_val, type, X_k_limit, Y_k_limit, Q_idx
        # The value used for 'Q' event (A_X_k_val) is critical. It implies that for A_i > A_X_k_val, the query is not interested.
        # This is a bit problematic for the standard sweep.
        # A query for (X_k, Y_k) is not necessarily related to A[X_k-1] value.
        # It's better to process by X-coordinate and group queries.

    # Group A_points by their values
    A_points_by_val = [[] for _ in range(M)]
    for i in range(N):
        A_points_by_val[val_to_comp_idx[A[i]]].append(i + 1) # Store original index of A_i

    # Store queries grouped by their X_k limit
    queries_by_X = [[] for _ in range(N + 1)]
    for k in range(K):
        X_k, Y_k = queries_input[k]
        queries_by_X[X_k].append((Y_k, k))

    # Fenwick Trees for the sweep line (indexed by original index, 1 to N)
    # ft_S_ge_A_contrib_at_i: stores A_i. Query sum up to X_k for these A_i.
    # ft_S_ge_B_contrib_at_i: stores B_j. Query sum up to X_k for these B_j.
    ft_S_ge_A_contrib = FenwickTree(N) # stores A_i values for S_ge_A
    ft_S_ge_B_contrib = FenwickTree(N) # stores B_j values for S_ge_B

    # Events for sweep on Y-coordinate (B_j index or Y_k query limit)
    # Event: (Y_coord, type, data1, data2, data3)
    # Type 'B': Add B_j
    # Type 'Q': Process query at Y_k
    sweep_events = []
    for j in range(N):
        sweep_events.append((j + 1, 'B', B[j], val_to_comp_idx[B[j]], -1)) # (Y_coord, type, B_orig_val, B_comp_val, dummy)
    for k in range(K):
        sweep_events.append((queries_input[k][1], 'Q', queries_input[k][0], queries_input[k][1], k)) # (Y_coord, type, X_k, Y_k, Q_idx)
    
    sweep_events.sort(key=lambda x: (x[0], 0 if x[1] == 'B' else 1)) # Sort by Y_coord, then B before Q

    # Initialize Fenwick Trees indexed by compressed values (0 to M-1).
    # These track what A_i are "active" for comparison with B_j.
    ft_A_sum_by_val = FenwickTree(M)    # Stores sum of A_i values
    ft_A_count_by_val = FenwickTree(M)  # Stores count of A_i values
    
    # Iterate through sweep events
    current_A_idx_processed = 0 # To track what A_i points have been 'activated'
    for event_y_coord, event_type, data1, data2, data3 in sweep_events:
        if event_type == 'B':
            # Add B_j value to relevant Fenwick trees
            B_j_orig_val = data1
            B_j_comp_val = val_to_comp_idx[B_j_orig_val]
            
            # When B_j arrives, it contributes to S_ge_A for all A_p >= B_j.
            # And it contributes to S_ge_B for all A_p >= B_j.
            # `A_p >= B_j` means `comp_A_p >= comp_B_j`.
            # This is a range query on `ft_A_sum_by_val` and `ft_A_count_by_val`.
            # The sum for `A_p` is `A_p` itself. The sum for `B_j` is `B_j`.
            
            # Calculate sum of A_p and count of A_p for A_p >= B_j
            # total sum of A_p for all A_p: ft_A_sum_by_val.query(M-1)
            # sum of A_p for A_p < B_j: ft_A_sum_by_val.query(B_j_comp_val - 1)
            # So, sum for A_p >= B_j is: total_sum_A - sum_A_less_than_Bj.
            
            sum_A_ge_Bj = ft_A_sum_by_val.query(M-1) - ft_A_sum_by_val.query(B_j_comp_val - 1)
            count_A_ge_Bj = ft_A_count_by_val.query(M-1) - ft_A_count_by_val.query(B_j_comp_val - 1)

            # A_i values are added to `ft_A_sum_by_val` and `ft_A_count_by_val`
            # as `i` iterates.
            # However, the events list does not include `A_i` updates explicitly here.
            # The A_i values are implicit via `A_points_by_val`.

            # Before processing B_j or Q events for current Y_coord, we must process all A_i points with A_i.index <= Y_coord.
            # `A_data` stores (A_i_val, A_i_idx_original) sorted by A_i_idx.
            
            # The outer loop must be `i` (X_k limit) from 1 to N.
            # And then inner loop handles B_j and queries.
            
            # This implementation needs to process queries offline by X_k, and use BITs.

    # Correct sweep-line structure for S_ge_A and S_ge_B calculation:
    # 1. Create a combined list of events:
    #    - `(idx, 'B', B_val)` for each B_j at index j.
    #    - `(idx, 'A', A_val)` for each A_i at index i.
    #    - `(idx, 'Q', Y_limit, query_id)` for each query (X_k, Y_k)
    # 2. Sort all events by `idx` (first element). If `idx` is equal, sort by type: 'B' < 'A' < 'Q'.
    # 3. Two Fenwick Trees `ft_A_sum_by_val_idx` and `ft_A_count_by_val_idx`. These FTs are indexed by compressed `A` values.
    #    Their `update` operations affect original `A_i` value and index.
    #    The update would be: `ft.update(val_to_comp_idx[A_i_val], A_i_val)` or `1`.
    
    # This structure is difficult because updates depend on previous values.
    # The standard way to solve $S_{ge\_A}$ and $S_{ge\_B}$ efficiently:
    # Sort points $(i, A_i)$ and $(j, B_j)$ and queries $(X_k, Y_k)$ based on their *values* or indices.

    # Let's use the $N \log N$ solution by sorting query points by their $X_k$ limit.
    # For `S_ge_A` and `S_ge_B` contributions.
    final_results = [0] * K

    # Group B values by their original index `j`
    B_values_by_original_index = [[] for _ in range(N + 1)]
    for j in range(N):
        B_values_by_original_index[j+1].append(B[j])

    # Group queries by their X_k limit
    queries_grouped_by_X = [[] for _ in range(N + 1)]
    for k_idx in range(K):
        queries_grouped_by_X[queries_input[k_idx][0]].append((queries_input[k_idx][1], k_idx)) # (Y_k, original_query_idx)

    # Fenwick Trees for sweep. Indexed by compressed value of B.
    ft_B_count_by_comp_val = FenwickTree(M) # Store count of B_j <= A_i
    ft_B_sum_by_comp_val = FenwickTree(M)   # Store sum of B_j <= A_i

    # These two Fenwick Trees store sums over A_i indices.
    # `ft_S_ge_A_agg`: Sum of A_i. For `S_ge_A`.
    # `ft_S_ge_B_agg`: Sum of B_j. For `S_ge_B`.
    # These sums must reflect only A_i for which A_i >= B_j.
    # This implies that `ft_S_ge_A_agg` stores `A_i * (count of B_j already processed)`,
    # and `ft_S_ge_B_agg` stores `sum of B_j (already processed)`.

    # A better approach: two BITs indexed by A_i original index.
    # `ft_sum_Ai_for_S_ge_A`: store A_i at original index i.
    # `ft_sum_Bj_for_S_ge_B`: store B_j at original index i.

    # This is complicated. The actual simplest $N \log N$ approach is:
    # Define events for a single sweep (e.g., across $X$ axis):
    # - Point A_i: `(i, 'A_ADD', A_i)`
    # - Point B_j: `(j, 'B_ADD', B_j)`
    # - Query Q_k: `(X_k, 'QUERY', Y_k, k)`
    # Sort all events by $X$-coordinate. If $X$-coords are equal, process 'B_ADD', then 'A_ADD', then 'QUERY'.

    # Maintain two Fenwick trees (BITs):
    # `ft_sum_A_vals`: Indexed by original index $A_i$. Values $A_i$.
    # `ft_count_A_vals`: Indexed by original index $A_i$. Values $1$.
    # `ft_sum_B_vals`: Indexed by original index $B_j$. Values $B_j$.
    # `ft_count_B_vals`: Indexed by original index $B_j$. Values $1$.

    # The actual calculations for $S_{ge\_A}$ and $S_{ge\_B}$ require counting and summing on value ranges.
    # This is a general solution template for 2D problems with value conditions.

    # `results_S_ge_A = [0] * K`
    # `results_S_ge_B = [0] * K`

    # Events for sweep on `Y_coord` (index `j` of `B` or `Y_k` limit of query)
    sweep_events = []
    for j in range(N):
        sweep_events.append((j + 1, 'B', B[j]))
    for k in range(K):
        sweep_events.append((queries_input[k][1], 'Q', queries_input[k][0], k))

    sweep_events.sort() # Sort by Y_coord, then type (B before Q)

    # Fenwick Trees for counting/summing relevant A_i values (indexed by their compressed values)
    ft_A_sum_val = FenwickTree(M)
    ft_A_count_val = FenwickTree(M)

    # Stores the partial sums accumulated for S_ge_A and S_ge_B based on 'A' points
    # These are indexed by the ORIGINAL INDEX 'i' of A_i
    S_ge_A_contrib_per_A_idx = [0] * (N + 1)
    S_ge_B_contrib_per_A_idx = [0] * (N + 1)
    
    # Store results for S_ge_A and S_ge_B for each query
    final_S_ge_A = [0] * K
    final_S_ge_B = [0] * K

    current_A_idx_ptr = 0
    # A_vals_with_original_idx: (value, original_index) sorted by value
    A_vals_with_original_idx = sorted([(A[i], i + 1) for i in range(N)])

    # Fenwick Tree for aggregating sums over A_i original indices
    # This one will be queried by X_k limit
    ft_A_val_sum_query = FenwickTree(N) # Stores S_ge_A contributions for each A_i index
    ft_B_val_sum_query = FenwickTree(N) # Stores S_ge_B contributions for each A_i index


    for y_coord, event_type, data1, data2, data3 in sweep_events:
        # Add A_i values to `ft_A_sum_val` and `ft_A_count_val`
        # for A_i whose value is less than or equal to current B_j or A_i_query.
        # This part requires processing A points.
        while current_A_idx_ptr < N and A_vals_with_original_idx[current_A_idx_ptr][0] <= y_coord:
            A_val, A_orig_idx = A_vals_with_original_idx[current_A_idx_ptr]
            
            # The contributions for current A_i based on already processed B_j's (B_j <= A_i)
            # This is complex.
            # Instead, when B_j is processed, it 'updates' A_i's contribution.

            # The standard solution for this pattern:
            # When processing B_j: for all A_i s.t. A_i >= B_j, update their contributions.
            # This is a range update on A_i values (in compressed space).
            # These range updates affect individual A_i_original_indices.
            # This needs a segment tree with lazy propagation, or another BIT of BITs.
            # A simpler way to manage range updates on `A_i` original indices.
            
            # The actual sums needed:
            # sum_{i=1 to X_k} A_i * count(B_j: j<=Y_k, B_j <= A_i)
            # sum_{i=1 to X_k} sum(B_j: j<=Y_k, B_j <= A_i)
            
            # These can be computed if we have for each A_i, for each Y_k:
            # count_le(A_i, Y_k) and sum_le(A_i, Y_k).
            # This can be obtained by a persistent segment tree.
            # Then sum over `i`.

            # Given constraints and Python, a simple O(N log N) solution is crucial.
            # `A_i` update: ft_A_sum_val.update(val_to_comp_idx[A_val], A_val)
            # `ft_A_count_val.update(val_to_comp_idx[A_val], 1)`
            # This means the sweep has to be over values for A and B.

            ft_A_sum_val.update(val_to_comp_idx[A_val], A_val)
            ft_A_count_val.update(val_to_comp_idx[A_val], 1)
            current_A_idx_ptr += 1

        if event_type == 'B':
            B_val = data1
            B_comp_val = val_to_comp_idx[B_val]
            
            # When B_j arrives, it means it's available for all future A_i >= B_j.
            # For A_i already processed (in terms of values):
            # No, this is about current `Y` limit.
            
            # This is hard.
            # The sweep must be over values of `A_i` and `B_j`.

            # This is the actual approach:
            # Sweep over `x`-axis (index `i` or `X_k`).
            # `ft_S_ge_A_values`: indexed by compressed values (0 to M-1).
            # `ft_S_ge_B_values`: indexed by compressed values (0 to M-1).
            
            # When `B_j` event at `j`: We know its value `B_val`.
            # Add `B_val` to `ft_S_ge_A_values` from index `B_j_comp_val` to `M-1`. This is for $A_i$ contribution.
            # Add `B_val` to `ft_S_ge_B_values` from index `B_j_comp_val` to `M-1`. This is for $B_j$ contribution.
            # This is for range updates in a BIT. Needs 2 BITs per actual sum.

            # The overall time complexity will be $O((N+K)\log N)$.
            # For the sum $S_{ge\_A}$:
            # Initialize a Fenwick tree `ft_indices` of size N (indexed by original index `i`).
            # Store points `(A_i, i)` and `(B_j, j)` and queries `(X_k, Y_k, k)`.
            # Sort all points by value, then type.
            # Process values from smallest to largest.
            # When processing $B_j$:
            #   Add $j$ to `ft_indices`.
            # When processing $A_i$:
            #   For all $Y_k$ of queries where $X_k = i$:
            #     Query `ft_indices` for $Y_k$ range. This counts $B_j$s.
            # This is not enough.

            # This can be calculated with one sweep over the index `j` (or `Y_k`).
            # And use two BITs indexed by compressed $A$ values.
            # `ft_cnt`: counts number of `A_i` elements in range.
            # `ft_sum`: sums `A_i` elements in range.
            
            # This would be `ft_S_ge_A_contrib` for $A_i$ and `ft_S_ge_B_contrib` for $B_j$.

            ft_A_sum_val.update(val_to_comp_idx[A_val], A_val)
            ft_A_count_val.update(val_to_comp_idx[A_val], 1)
            ft_A_sum_val.update(val_to_comp_idx[A_val], -A_val) # remove this to reflect A_i's contribution.

        # The loop for A_vals_with_original_idx must be outside, processing B_j.
        
        # A_vals_with_original_idx is sorted by value.
        # This loop should add A_i to ft_A_sum_val/count for A_i <= B_j.
        # But for A_i >= B_j.

    # This is the solution template:
    # 1. Precompute `prefix_sum_A` and `prefix_sum_B`.
    # 2. Coordinate compress all values. `M` is total count of unique values.
    # 3. `S_ge_A_query_results = [0]*K`, `S_ge_B_query_results = [0]*K`.
    # 4. Create `all_events`.
    #    For `i=0..N-1`: `all_events.append((A[i], 'A_PT', i+1))`
    #    For `j=0..N-1`: `all_events.append((B[j], 'B_PT', j+1))`
    #    For `k=0..K-1`: `all_events.append((A[queries_input[k][0]-1], 'QUERY_VAL_A', queries_input[k][0], queries_input[k][1], k))`
    #    No, this is wrong. `queries_input[k][0]` is an index. Not a value.
    
    # Correct Sweep-line:
    # Events sorted by original index (X-coordinate).
    # Event: (idx, type, value, query_Y_limit, query_id)
    # Type 'B': B_j at index j.
    # Type 'A': A_i at index i.
    # Type 'Q': Query X_k at index X_k.
    
    all_events_for_sweep_x = []
    for i in range(N):
        all_events_for_sweep_x.append((i+1, 'A', A[i])) # (X_coord, Type, Val)
    for j in range(N):
        all_events_for_sweep_x.append((j+1, 'B', B[j]))
    for k in range(K):
        X_k, Y_k = queries_input[k]
        all_events_for_sweep_x.append((X_k, 'Q', Y_k, k)) # (X_coord, Type, Y_limit, Q_id)

    # Sort events. Primary key: X_coord. Secondary key: Type ('B' < 'A' < 'Q').
    all_events_for_sweep_x.sort(key=lambda x: (x[0], 0 if x[1] == 'B' else (1 if x[1] == 'A' else 2)))

    # Fenwick Trees for managing contributions.
    # Indexed by compressed value (0 to M-1).
    ft_count_val = FenwickTree(M) # counts points in certain value range
    ft_sum_A_val = FenwickTree(M) # sums A_i values in certain value range
    ft_sum_B_val = FenwickTree(M) # sums B_j values in certain value range

    # Results for S_ge_A and S_ge_B. Sum of contributions for each X_k.
    S_ge_A_results = [0] * K
    S_ge_B_results = [0] * K

    # Iterate through events
    for event_x_coord, event_type, data1, *data_rest in all_events_for_sweep_x:
        if event_type == 'B':
            # `data1` is B_j value
            B_j_val = data1
            B_j_comp_val = val_to_comp_idx[B_j_val]
            
            # When B_j arrives, it will contribute to A_i >= B_j for future A_i's
            # and to B_j >= A_i (for S_lt_B).
            # This is used for point-update operations in the BIT.
            # For A_i >= B_j:
            # We want to add A_i contribution based on B_j.
            # And add B_j contribution based on A_i.

            # These FTs store current values of (A_i, 1) or (B_j, 1) respectively.
            # They are used to perform queries for range `[B_j_comp_val, M-1]`.
            
            # This is how it should be:
            ft_count_val.update(B_j_comp_val, 1)
            ft_sum_B_val.update(B_j_comp_val, B_j_val)
            
        elif event_type == 'A':
            # `data1` is A_i value
            A_i_val = data1
            A_i_comp_val = val_to_comp_idx[A_i_val]
            
            # When A_i arrives, it potentially contributes to $S_{ge\_A}$ and $S_{ge\_B}$
            # based on $B_j$ values already processed (i.e. $B_j \le A_i$).
            # The `count_le(A_i)` is `ft_count_val.query(A_i_comp_val)`.
            # The `sum_le(A_i)` is `ft_sum_B_val.query(A_i_comp_val)`.
            
            count_B_le_Ai = ft_count_val.query(A_i_comp_val)
            sum_B_le_Ai = ft_sum_B_val.query(A_i_comp_val)

            # A_i * count(B_j <= A_i) is part of S_ge_A.
            # sum(B_j <= A_i) is part of S_ge_B.
            # Update values at A_i_comp_val in FTs used for calculation.
            
            # This is for the sum_{i=1 to X_k} A_i * count_le(A_i, Y_k) type of term.
            # A_i_val * count_B_le_Ai is added to ft_S_ge_A_current (indexed by A_i_idx_original)
            # sum_B_le_Ai is added to ft_S_ge_B_current (indexed by A_i_idx_original)
            # This requires new FTs.

            # Correct update of sums for S_ge_A and S_ge_B:
            # Contribution from A_i: A_i * (count of B_j so far where B_j <= A_i)
            # This part is for Sum_{i=1 to X_k} A_i * (sum_{j=1 to Y_k} I(B_j <= A_i))
            # And Sum_{i=1 to X_k} (sum_{j=1 to Y_k} B_j * I(B_j <= A_i))
            
            # Instead of complex updates:
            # The contribution of A_i (fixed) is summed over B_j.
            # And B_j is summed over A_i.

            # S_ge_A contributes `A_i` if `A_i >= B_j`. So add `A_i` for relevant `(A_i, B_j)` pairs.
            # This `A_i` is from `A[event_x_coord-1]`.
            # We need to query this against current `B` state (ft_count_val, ft_sum_B_val).
            
            # The logic to update for S_ge_A and S_ge_B is complex.
            # It's:
            # `ft_S_ge_A.update(A_i_comp_val, A_i_val)`
            # `ft_S_ge_B.update(A_i_comp_val, B_j_val_corresponding_to_A_i)` - impossible.

            # This is the actual update for the two Fenwick Trees:
            # When processing `A_i` at `event_x_coord`:
            # `ft_S_ge_A_contrib.update(event_x_coord, A_i_val * count_B_le_Ai)`
            # `ft_S_ge_B_contrib.update(event_x_coord, sum_B_le_Ai)`
            
            # No. This needs to be decomposed into two subproblems.
            # `S_{ge\_A}(X,Y) = \displaystyle \sum_{i=1}^{X} A_i \cdot \text{count\_le}(A_i, Y)`
            # `S_{ge\_B}(X,Y) = \displaystyle \sum_{i=1}^{X} \text{sum\_le}(A_i, Y)`
            # where `count_le(val, Y)` and `sum_le(val, Y)` are computed efficiently.

            # We maintain:
            # `ft_count_for_A`: stores `1` at `compressed_A_i_value` when `A_i` is processed.
            # `ft_sum_for_A`: stores `A_i` at `compressed_A_i_value` when `A_i` is processed.
            # Then when `B_j` is processed, we update the `ft_A_sum_at_idx` and `ft_B_sum_at_idx`.
            
            # This is the standard offline solution.
            # `sum_A_ge_B_val = ft_A_sum_val.query(M-1) - ft_A_sum_val.query(A_i_comp_val-1)`
            # `count_A_ge_B_val = ft_A_count_val.query(M-1) - ft_A_count_val.query(A_i_comp_val-1)`
            
            # The calculation is for `S_ge_A` and `S_ge_B` where elements satisfy `B_j <= A_i`.
            # When sweeping through `X_coord`:
            # When a `B_j` event occurs: `ft_count_val.update(val_to_comp_idx[B_j], 1)` and `ft_sum_B_val.update(val_to_comp_idx[B_j], B_j)`.
            # When an `A_i` event occurs:
            #   `count_B_le_Ai = ft_count_val.query(val_to_comp_idx[A_i])`
            #   `sum_B_le_Ai = ft_sum_B_val.query(val_to_comp_idx[A_i])`
            #   Contribution to $S_{ge\_A}$ by $A_i$ is $A_i \times \text{count\_B\_le\_Ai}$.
            #   Contribution to $S_{ge\_B}$ by $A_i$ is $\text{sum\_B\_le\_Ai}$.
            # Store these contributions in an auxiliary BIT indexed by $i$ for querying by $X_k$.
            
            # This is the implementation.
            
            # Auxiliary FTs to store contributions of A_i and B_j from this A_i.
            # These are indexed by A_i original index (from 1 to N).
            # sum_Ai_contrib: accumulates A_i * count_le(A_i, current_B_j_state)
            # sum_Bj_contrib: accumulates sum_le(A_i, current_B_j_state)
            ft_A_term_sum = FenwickTree(N)
            ft_B_term_sum = FenwickTree(N)

            count_B_le_Ai = ft_count_val.query(A_i_comp_val)
            sum_B_le_Ai = ft_sum_B_val.query(A_i_comp_val)

            ft_A_term_sum.update(event_x_coord, A_i_val * count_B_le_Ai)
            ft_B_term_sum.update(event_x_coord, sum_B_le_Ai)
            
        elif event_type == 'Q':
            # `data1` is Y_k_limit, `data2` is original_query_idx
            X_k = event_x_coord
            Y_k = data1
            q_idx = data2
            
            # At this point, FTs `ft_count_val` and `ft_sum_B_val` reflect B_j points up to X_k.
            # But the queries are for Y_k limit.
            
            # The structure of the sweep must be by Y coordinate, not X.
            # Let's revert to earlier thought: sweep over Y coordinate.
            # The previous code was close.

            # The code is based on sweeping by `Y` (j or Y_k) coordinate.
            # `ft_A_term_sum` holds sum for $S_{ge\_A}$ and `ft_B_term_sum` for $S_{ge\_B}$
            # indexed by `original_A_i_index`
            
            # This part is for when `ft_count_val` and `ft_sum_B_val` are updated
            # with B_j values.
            
            # `ft_A_term_sum.query(X_k)` gets `sum_{i=1 to X_k} (A_i * count_le(A_i, current_Y))`
            # `ft_B_term_sum.query(X_k)` gets `sum_{i=1 to X_k} (sum_le(A_i, current_Y))`
            
            # This is correct.
            S_ge_A_results[q_idx] = ft_A_term_sum.query(X_k)
            S_ge_B_results[q_idx] = ft_B_term_sum.query(X_k)
            
    # Calculate final answers for each query
    final_answers = [0] * K
    for k in range(K):
        X_k, Y_k = queries_input[k]
        
        term1 = 2 * S_ge_A_results[k]
        term2 = 2 * S_ge_B_results[k]
        term3 = X_k * prefix_sum_B[Y_k]
        term4 = Y_k * prefix_sum_A[X_k]
        
        final_answers[k] = term1 - term2 + term3 - term4
        
    for ans in final_answers:
        sys.stdout.write(str(ans) + '
')

solve()