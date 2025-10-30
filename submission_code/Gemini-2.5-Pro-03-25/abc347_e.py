# YOUR CODE HERE
import sys

# --- Main logic ---
def solve():
    # Read N (length of sequence A) and Q (number of queries)
    N, Q = map(int, sys.stdin.readline().split())
    # Read the sequence of queries X
    X = list(map(int, sys.stdin.readline().split()))

    # Pass 1: Compute the size of set S after each query.
    # We use a list `sizes` where sizes[i] will store the size of S *after* processing the i-th query (1-based index).
    # `sizes[0]` is unused or can be considered 0.
    sizes = [0] * (Q + 1) 
    # `current_S` keeps track of the elements currently in set S during this pass.
    current_S = set()
    
    # Iterate through each query from 1 to Q.
    for i in range(Q):
        # Get the element value for the current query (i-th query in 0-based index, corresponds to query i+1).
        x_i = X[i]
        
        # Update the set S based on whether x_i is already present.
        if x_i in current_S:
            # If x_i is in S, remove it.
            current_S.remove(x_i)
        else:
            # If x_i is not in S, insert it.
            current_S.add(x_i)
            
        # Record the size of S after processing the (i+1)-th query.
        # Store it at index i+1 in the `sizes` list.
        sizes[i+1] = len(current_S) 

    # Pass 2: Compute prefix sums of the `sizes` array.
    # `P[k]` will store the cumulative sum of sizes from query 1 up to query k.
    # This allows calculating the sum of sizes over any interval [start, end] in O(1) time.
    P = [0] * (Q + 1)
    for i in range(1, Q + 1):
        P[i] = P[i-1] + sizes[i]

    # Pass 3: Compute the final values for array A.
    # Initialize the result array A with N zeros. A is 0-indexed, so A[k-1] corresponds to element k.
    A = [0] * N 
    
    # We need to re-simulate the set updates to determine the intervals during which each element is present in S.
    # `current_S_pass3` tracks the set S state during this pass.
    current_S_pass3 = set()
    # `last_insert_time` is a dictionary mapping an element k (1-based value) to the query index (1-based)
    # when it was most recently inserted into the set S.
    last_insert_time = {} 

    # Iterate through each query again.
    for i in range(Q):
        # Current query index is i+1 (1-based).
        query_idx = i + 1 
        # Current query element value.
        x_i = X[i] 
        
        # Check if x_i is currently in the set being tracked in this pass.
        if x_i in current_S_pass3:
            # If yes, this query removes x_i from the set.
            current_S_pass3.remove(x_i)
            
            # Retrieve the query index when x_i was last inserted.
            start_time = last_insert_time[x_i] 
            # The query index when x_i is removed is the current query index.
            end_time = query_idx 
            
            # The element x_i was present in the actual set S during the interval of queries [start_time, end_time - 1].
            # According to the problem statement, for each query p in this interval, we added |S^{(p)}| (which is sizes[p]) to A[x_i-1].
            # The total value added to A[x_i-1] during this interval is the sum of sizes[p] for p from start_time to end_time - 1.
            # This sum can be computed efficiently using the precomputed prefix sums P: Sum = P[end_time - 1] - P[start_time - 1].
            interval_sum = P[end_time - 1] - P[start_time - 1]
            
            # Add this calculated sum to the corresponding element in A (using 0-based index x_i - 1).
            A[x_i - 1] += interval_sum
            
            # Since x_i is removed, remove its entry from the tracking dictionary.
            del last_insert_time[x_i]
        else:
            # If x_i is not in the set, this query inserts it.
            current_S_pass3.add(x_i)
            # Record the query index of this insertion event for x_i.
            last_insert_time[x_i] = query_idx

    # After processing all Q queries, some elements might still remain in the set `current_S_pass3`.
    # These elements were present in S from their `last_insert_time` until the very end (after query Q).
    # For each such element k still in the set:
    for k in current_S_pass3:
        # Retrieve the query index when k was last inserted.
        start_time = last_insert_time[k]
        # The element k was present in S for queries from start_time up to Q inclusive.
        # The total value added to A[k-1] during this final interval is the sum of sizes[p] for p from start_time to Q.
        # Using prefix sums, this sum is P[Q] - P[start_time - 1].
        interval_sum = P[Q] - P[start_time - 1]
        
        # Add this sum to the corresponding element in A (using 0-based index k - 1).
        A[k - 1] += interval_sum

    # Print the final state of array A. Elements should be space-separated.
    # The `*` operator unpacks the list A into individual arguments for print.
    print(*(A))

# --- Calling the main logic ---
# The `solve()` function contains the entire logic and will be executed when the script runs.
solve()