import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Store queries and collect all distinct values for coordinate compression
    queries_input = []
    all_values_set = set() 
    for x in A:
        all_values_set.add(x)

    for i in range(Q):
        R_val, X_val = map(int, sys.stdin.readline().split())
        # R_val is 1-indexed in problem, corresponds to A[0...R_val-1]
        # Store R_val-1 as R_idx for 0-indexed processing
        queries_input.append({'R_idx': R_val - 1, 'X': X_val, 'original_idx': i})
        all_values_set.add(X_val)

    # Coordinate compression
    # sorted_unique_values contains all unique values from A and X_i, sorted.
    sorted_unique_values = sorted(list(all_values_set))
    # val_to_compressed maps each unique value to a 1-based rank.
    val_to_compressed = {val: i + 1 for i, val in enumerate(sorted_unique_values)}
    
    # M is the number of unique values, also the max compressed value.
    # BIT will be of size M+1 for 1-based indexing up to M.
    M = len(sorted_unique_values) 
    
    bit = [0] * (M + 1) # Initialize BIT with zeros

    # Updates the BIT: sets the LIS length for val_idx to be at least val.
    # idx is 1-based compressed value.
    # val is the LIS length.
    def update_bit(idx, val):
        while idx <= M:
            bit[idx] = max(bit[idx], val)
            idx += idx & (-idx) 

    # Queries the BIT: gets max LIS length for elements with compressed value <= idx.
    # idx is 1-based compressed value.
    def query_bit(idx):
        max_val = 0
        while idx > 0:
            max_val = max(max_val, bit[idx])
            idx -= idx & (-idx) 
        return max_val

    # Group queries by their R_idx (0-indexed R)
    # queries_by_R_idx[i] stores list of (compressed_X, original_query_idx)
    # for all queries that concern A[0...i]
    queries_by_R_idx = [[] for _ in range(N)]
    for query_info in queries_input:
        compressed_X = val_to_compressed[query_info['X']]
        queries_by_R_idx[query_info['R_idx']].append(
            (compressed_X, query_info['original_idx'])
        )
    
    ans = [0] * Q # Stores answers in original query order

    # Iterate through the array A, processing A[i] at step i
    for i in range(N): # i is 0-indexed, A[i] is the (i+1)-th element in 1-based problem statement
        val_A = A[i]
        compressed_A = val_to_compressed[val_A]
        
        # Length of LIS ending in val_A is 1 + (max LIS length ending in some val_A_prev < val_A)
        # Query for max LIS length for values strictly less than val_A (i.e., up to compressed_A - 1).
        # If compressed_A is 1 (smallest value overall), compressed_A - 1 is 0. query_bit(0) returns 0.
        len_prev = query_bit(compressed_A - 1)
        current_lis_len = len_prev + 1
        
        # Update BIT: an LIS of length current_lis_len ending with val_A is found.
        update_bit(compressed_A, current_lis_len)
        
        # Process all queries that have R_idx = i (i.e., R = i+1 in problem statement)
        for compressed_X, original_idx in queries_by_R_idx[i]:
            # Query for max LIS length for values <= X (i.e., with compressed value <= compressed_X)
            # This is exactly what query_bit(compressed_X) provides.
            query_ans = query_bit(compressed_X)
            ans[original_idx] = query_ans
            
    for i in range(Q):
        sys.stdout.write(str(ans[i]) + "
")

if __name__ == '__main__':
    main()