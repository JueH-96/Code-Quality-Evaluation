import sys

# Increase recursion depth for safety, although the solution is iterative.
# sys.setrecursionlimit(200000) 

def solve():
    """
    Solves the problem based on the described logic:
    1. Calculate the initial inversion count I(0) using a Fenwick tree (BIT).
    2. Precompute the change in inversion count delta(v) for each possible value v.
       The change occurs when elements with value v = (M-1-k) mod M wrap around.
    3. Calculate I(k) for k=1..M-1 iteratively using the recurrence I(k+1) = I(k) + delta_k.
    """
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Fenwick tree (BIT) implementation
    # The BIT stores counts for values 0 to M-1.
    # Internally, it uses 1-based indexing. Array size needs to be M+1.
    # Index `j` in the BIT corresponds to value `j-1`.
    M_sz = M + 1 
    bit = [0] * M_sz

    def update(idx, val):
        """ 
        Updates the BIT by adding 'val' to the count associated with value 'idx' (0-based). 
        Maps 0-based value `idx` to 1-based BIT index `idx + 1`.
        """
        idx += 1 
        while idx < M_sz:
            bit[idx] += val
            idx += idx & (-idx) # Move to the next index to update

    def query_prefix(idx):
        """ 
        Queries the cumulative count for values from 0 up to 'idx' (0-based, inclusive). 
        Maps 0-based value `idx` to 1-based BIT index `idx + 1`.
        """
        idx += 1 
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & (-idx) # Move to the parent index
        return s

    # Calculate initial inversion count I(0) for sequence A.
    # Since 0 <= A_i < M, A_i mod M = A_i. So B(0) = A.
    inv0 = 0
    # Iterate through the sequence A
    for i in range(N):
        val = A[i] # Current value
        
        # We need to count elements A[j] already processed (j < i) such that A[j] > val.
        # These elements have values in the range [val + 1, M - 1].
        # The count can be found using BIT prefix sums:
        # Count(values > val) = TotalCount - Count(values <= val)
        
        # Total count of elements processed so far is query_prefix(M-1)
        # Count of elements with value <= val is query_prefix(val)
        count_greater = query_prefix(M - 1) - query_prefix(val)
        
        # Add the count of inversions involving the current element A[i]
        inv0 += count_greater
        
        # Mark the current value 'val' as seen by incrementing its count in the BIT.
        update(val, 1)

    # Precompute locations (1-based indices) of each value in A
    indices_by_value = [[] for _ in range(M)]
    for i in range(N):
        # Store 1-based index (i+1) because the problem uses 1-based indexing in its definition.
        indices_by_value[A[i]].append(i + 1) 

    # Precompute the change in inversion count delta(v) associated with each value v.
    # The change delta_k depends on the set S_k = {i | A_i = (M-1-k) mod M}.
    # The formula derived for the change I(k+1) - I(k) is sum_{i in S_k} (2*i - N - 1).
    # We precompute this sum for each possible value v.
    changes = [0] * M
    for v in range(M):
        current_change = 0
        # Sum the formula over all 1-based indices `idx` where A_{idx-1} = v
        for idx in indices_by_value[v]: 
            current_change += (2 * idx - N - 1)
        changes[v] = current_change

    # Compute results for k=0..M-1 using the recurrence relation
    results = [0] * M
    results[0] = inv0 # Base case: I(0)
    current_inv = inv0

    # Iterate k from 0 to M-2 to find I(1) through I(M-1)
    for k in range(M - 1):
        # Determine the value `v` whose corresponding elements wrap around from M-1 to 0
        # when k increments to k+1. These are elements A_i such that A_i + k = M-1 (mod M).
        # This condition means A_i = (M - 1 - k) mod M.
        # Calculate (M - 1 - k) ensuring it's non-negative before modulo operation.
        val_to_find = (M - 1 - k + M) % M 
        
        # Retrieve the precomputed change associated with this value `v`
        delta = changes[val_to_find]
        
        # Update the inversion count using the recurrence relation
        current_inv += delta
        results[k+1] = current_inv # Store I(k+1)

    # Print the computed inversion counts for k = 0 to M-1
    for res in results:
        print(res)

# Execute the solve function
solve()